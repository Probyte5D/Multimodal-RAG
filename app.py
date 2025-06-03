import os
import torch
try:
    from deep_translator import GoogleTranslator
    has_translator = True
except ModuleNotFoundError:
    has_translator = False
    print("⚠️ deep-translator non è installato. La traduzione verrà saltata.")


# Fix temporaneo per errore RuntimeError su torch.classes.__path__
try:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
except Exception:
    torch.classes.__path__ = []

import nest_asyncio
nest_asyncio.apply()

import asyncio
import sys
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from PIL import Image


from models.blip_model import extract_caption
from models.vector_store import (
    init_milvus_collection,
    get_embedding,
    get_image_id,
    insert_to_milvus,
    search_similar
)
from models.gpt_model import generate_response_stream
from models.image_embedder import ImageEmbedder
from models.utils import process_amazon_folder  # batch function

# Inizializza modello di embedding e Milvus
image_embedder = ImageEmbedder()
collection = init_milvus_collection()

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG")

lang = st.radio("🌍 Lingua della descrizione e delle risposte", ["it", "en"], index=0)

amazon_folder = "/app/images"

# Pulsante per processare immagini da una cartella
if st.button("Processa immagini Amazon"):
    with st.spinner("Sto processando le immagini Amazon..."):
        process_amazon_folder(amazon_folder, image_embedder, collection)
    st.success("✅ Immagini Amazon processate!")

uploaded_file = st.file_uploader("Upload un'immagine", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Immagine caricata", use_container_width=True)

    image_embedding_list = image_embedder.encode_image([image])

    if not image_embedding_list or image_embedding_list[0] is None:
        st.error("❌ Errore: embedding non generato. Verifica il modello.")
        st.stop()

    image_embedding = image_embedding_list[0]

    import numpy as np
    if hasattr(image_embedding, "detach"):
        image_embedding = image_embedding.detach().cpu().numpy()
    if isinstance(image_embedding, np.ndarray):
        image_embedding = image_embedding.astype(float).tolist()

    if not isinstance(image_embedding, list) or not all(isinstance(x, (float, int)) for x in image_embedding):
        st.error("❌ Errore: embedding immagine non è una lista valida di float")
        st.stop()

    with st.expander("🔍 Dettagli tecnici embedding"):
        st.write("Tipo embedding:", type(image_embedding).__name__)
        st.write("Preview embedding:", image_embedding[:5])

    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    image_id = get_image_id(image_bytes)

    with st.spinner("Estrazione descrizione..."):
        caption = extract_caption(image)

    # Traduzione automatica in italiano se lingua sospetta (inglese)
    if has_translator and caption and caption.strip() and any(word in caption.lower() for word in [" a ", " the ", "with ", " on ", " of "]):
        caption = GoogleTranslator(source='auto', target='it').translate(caption)


    st.markdown("### 📷 Descrizione Base")
    st.markdown(f"> {caption}")

    # Inserisci embeddings in Milvus
    caption_embedding = get_embedding(caption)
    insert_to_milvus(collection, caption_embedding, image_embedding, caption, image_id)

    # Cerca descrizioni e immagini simili
    similar_texts = search_similar(collection, caption_embedding, anns_field="text_embedding", exclude_image_id=image_id)
    similar_images = search_similar(collection, image_embedding, anns_field="image_embedding", exclude_image_id=image_id)

    # Contesto completo (testi da similar_texts + similar_images senza duplicati)
    all_contexts = list(dict.fromkeys([txt for txt, _ in (similar_texts + similar_images)]))

    st.markdown("### 🧐 Descrizioni simili trovate")
    for i, (txt, _) in enumerate(similar_texts):
        st.markdown(f"{i+1}. {txt}")

    improved_caption = ""

    def build_caption_prompt(base_caption, similar_descs=None):
        if similar_descs:
            desc_context = "\n".join(f"- {desc}" for desc in similar_descs)
            return f"""Questa è una descrizione automatica dell'immagine: "{base_caption}"

Ecco alcune descrizioni simili da immagini precedenti:
{desc_context}

📌 Obiettivo: fornire una **descrizione migliorata**, **basata solo sugli elementi visibili** nell'immagine.
Linee guida:
- Includi oggetti presenti, quantità, colore e disposizione.
- Se ci sono testi o simboli, trascrivili esattamente.
- Se ci sono sezioni multiple, descrivile singolarmente.
- Se alcuni oggetti sono sfocati, tagliati o oscurati, segnalalo in modo chiaro.
- Evita interpretazioni o aggiunte non evidenti.

Scrivi una descrizione visiva dettagliata e coerente:"""
        else:
            return f"""Questa è una descrizione automatica dell'immagine: "{base_caption}"

Scrivi una descrizione visiva migliorata, **basandoti solo sugli elementi visibili** nell'immagine.
Indica gli oggetti presenti, la quantità, i colori e la disposizione.
Se ci sono testi o simboli, trascrivili esattamente.
Se alcuni oggetti sono sfocati, tagliati o difficili da vedere, segnalalo.
Evita aggiunte non supportate dall'immagine.

La descrizione deve essere chiara, concisa e informativa:"""

    if similar_texts:
        with st.spinner("📚 Miglioramento descrizione..."):
            similar_descriptions = [txt for txt, _ in similar_texts if isinstance(txt, str) and txt.strip()]

            if not similar_descriptions:
                st.warning("⚠️ Nessuna descrizione simile valida trovata. Uso solo la descrizione base.")
                context_input = [caption]
                caption_prompt = build_caption_prompt(caption)
            else:
                context_input = similar_descriptions
                caption_prompt = build_caption_prompt(caption, similar_descriptions)

            with st.expander("📷 ✨ Descrizione migliorata", expanded=True):
                improved_caption_placeholder = st.empty()
                for token in generate_response_stream(
                    context=context_input,
                    user_input=caption_prompt,
                    lang=lang
                ):
                    improved_caption += token
                    improved_caption_placeholder.markdown(improved_caption)
    else:
        with st.spinner("📚 Nessuna descrizione simile trovata, miglioro quella base..."):
            caption_prompt = build_caption_prompt(caption)

            with st.expander("📷 ✨ Descrizione migliorata", expanded=True):
                improved_caption_placeholder = st.empty()
                for token in generate_response_stream(
                    context=[caption],
                    user_input=caption_prompt,
                    lang=lang
                ):
                    improved_caption += token
                    improved_caption_placeholder.markdown(improved_caption)

    st.markdown("---")
    st.markdown("## 💬 Fai una domanda sull'immagine")

    user_question = st.text_input("Scrivi la tua domanda qui")

    if user_question:
        combined_context = [caption] + [t[0] for t in similar_texts if t[0] != caption] if similar_texts else [caption]
        answer = ""
        with st.spinner("💡 Sto analizzando l'immagine..."):
            with st.expander("📌 Risposta alla tua domanda", expanded=True):
                user_response_placeholder = st.empty()
                for token in generate_response_stream(
                    context=combined_context,
                    user_input=user_question,
                    lang=lang
                ):
                    answer += token
                    user_response_placeholder.markdown(answer)
