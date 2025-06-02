import os
import torch

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

st.set_page_config(page_title="Image to Text RAG con Milvus", layout="wide")
st.title("Image to Text RAG con Milvus")

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


    # ✅ Conversione in lista di float (se necessario)
    import numpy as np
    if hasattr(image_embedding, "detach"):
        image_embedding = image_embedding.detach().cpu().numpy()
    if isinstance(image_embedding, np.ndarray):
        image_embedding = image_embedding.astype(float).tolist()

    # ✅ Validazione
    if not isinstance(image_embedding, list) or not all(isinstance(x, float) or isinstance(x, int) for x in image_embedding):
        st.error("❌ Errore: embedding immagine non è una lista valida di float")
        st.stop()

    # Debug info
    st.write("🔍 Tipo embedding:", type(image_embedding))
    st.write("🔍 Preview embedding:", image_embedding[:5])


    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    image_id = get_image_id(image_bytes)

    with st.spinner("Estrazione descrizione..."):
        caption = extract_caption(image)

    st.markdown("### 📷 Descrizione Base")
    st.markdown(f"> {caption}")

    # Embedding testo e inserimento in Milvus
    caption_embedding = get_embedding(caption)
    insert_to_milvus(collection, caption_embedding, image_embedding, caption, image_id)

    # Recupera testi e immagini simili
    similar_texts = search_similar(collection, caption_embedding, anns_field="text_embedding", exclude_image_id=image_id)
    similar_images = search_similar(collection, image_embedding, anns_field="image_embedding", exclude_image_id=image_id)

    # Prepara contesto completo (senza duplicati)
    all_contexts = [txt for txt, _ in (similar_texts + similar_images)]
    all_contexts = list(dict.fromkeys(all_contexts))

    st.markdown("### 🧠 Descrizioni simili trovate")
    for i, (txt, _) in enumerate(similar_texts):
        st.markdown(f"{i+1}. {txt}")

    # Caption migliorata usando GPT
    if similar_texts:
        with st.spinner("📚 Miglioramento descrizione..."):
            caption_context = "\n".join([t[0] for t in similar_texts])
            caption_prompt = f"""Questa è una descrizione automatica dell'immagine: "{caption}"

Descrizioni simili da immagini precedenti:
{caption_context}

Analizza attentamente il contenuto visivo descritto.
- Se ci sono più oggetti, elencali con quantità e disposizione.
- Se è presente testo visibile o simboli, includili nella descrizione.
- Se l'immagine è un collage o una griglia, descrivi ogni sezione.
- Se alcuni oggetti sono parziali o sfocati, menzionalo.

Fornisci una **descrizione migliorata**, dettagliata ma concisa:"""

            response_placeholder = st.empty()
            improved_caption = ""
            for token in generate_response_stream([caption_context], caption_prompt, lang="it"):
                improved_caption += token
                response_placeholder.markdown(f"### ✨ Descrizione migliorata\n{improved_caption}")

    # Domanda personalizzata dell'utente
    user_question = st.text_input("Fai una domanda sull'immagine")

    if user_question:
        combined_context = [caption] + [t[0] for t in similar_texts if t[0] != caption]
        response_placeholder = st.empty()
        answer = ""
        for token in generate_response_stream(combined_context, user_question, lang="it"):
            answer += token
            response_placeholder.markdown(answer)
