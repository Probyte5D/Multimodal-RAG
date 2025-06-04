import os
from glob import glob
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from sklearn.cluster import KMeans

from models.blip_model import extract_caption
from models.vector_store import get_embedding, insert_to_milvus, get_image_id


class ImageEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, images: list[Image.Image]) -> list[list[float]]:
        embeddings = []
        for img in images:
            try:
                inputs = self.processor(images=img, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = self.model.get_image_features(**inputs)
                    normed = output / output.norm(dim=-1, keepdim=True)
                    embeddings.append(normed.cpu().numpy()[0].tolist())
            except Exception as e:
                print(f"Errore durante l'embedding dell'immagine: {e}")
                embeddings.append(None)
        return embeddings


def cluster_embeddings(embeddings, n_clusters=10):
    print("⏳ Clustering embeddings...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(np.array(embeddings))
    print("✅ Clustering completato.")
    return clusters


def process_amazon_folder(image_folder, image_embedder, collection, batch_size=8, img_size=160, n_clusters=10):
    image_paths = glob(os.path.join(image_folder, "*.jpg"))

    all_images = []
    all_paths = []

    print("🔍 Caricamento immagini e calcolo embeddings...")
    for path in tqdm(image_paths):
        try:
            img = Image.open(path).convert("RGB").resize((img_size, img_size))
            all_images.append(img)
            all_paths.append(path)
        except Exception as e:
            print(f"Errore caricamento {path}: {e}")

    embeddings = []
    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i + batch_size]
        embs = image_embedder.encode_image(batch)
        embeddings.extend(embs)

    clusters = cluster_embeddings(embeddings, n_clusters=n_clusters)

    for idx, (img, emb, path, cluster_id) in enumerate(zip(all_images, embeddings, all_paths, clusters)):
        if emb is None:
            print(f"❌ Embedding mancante per immagine {path}, saltata.")
            continue

        try:
            with open(path, "rb") as f:
                image_bytes = f.read()
            image_id = get_image_id(image_bytes)
        except Exception as e:
            print(f"Errore lettura {path}: {e}")
            continue

        caption = extract_caption(img)
        text_emb = get_embedding(caption)
        image_emb = np.array(emb, dtype=np.float32)
        text_emb = np.array(text_emb, dtype=np.float32)

        partition_name = f"cluster_{cluster_id}"
        if not collection.has_partition(partition_name):
            collection.create_partition(partition_name)

        insert_to_milvus(
            collection,
            text_emb,
            image_emb,
            caption,
            image_id,
            partition_name=partition_name
        )

    print("🏁 Inserimento immagini completato.")

def get_existing_image_ids(collection, image_ids: list[str]) -> set:
    filtered_ids = [id_ for id_ in image_ids if id_ is not None]
    if not filtered_ids:
        return set()
    expr = f'image_id in {filtered_ids}'
    try:
        results = collection.query(expr=expr, output_fields=["image_id"])
        return set(r["image_id"] for r in results)
    except Exception as e:
        print(f"❌ Errore durante la query per image_id esistenti: {e}")
        return set()


if __name__ == "__main__":
    collection = init_milvus_collection()
    image_embedder = ImageEmbedder()
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    image_folder = os.path.join(base_path, "images_folder", "images")
    print("🔎 Path immagini:", image_folder)
    print("📂 Contenuto:", os.listdir(image_folder))
    image_paths = glob(os.path.join(image_folder, "*.jpg"))
    print(f"🖼️ Immagini trovate: {len(image_paths)}")


    # Avvia il processo di caricamento, embedding e inserimento
    process_amazon_folder(image_folder, image_embedder, collection)

    # Esempio di ricerca testo
    query = "un cane che gioca sulla spiaggia"
    results = search_similar_by_text(collection, query)

    print("\n🔎 Risultati:")
    for r in results:
        print(f"- {r['caption']} (image_id={r['image_id']}, distanza={r['distance']:.4f})")
