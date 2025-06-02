import os
from glob import glob
from PIL import Image
from tqdm import tqdm

from models.vector_store import get_embedding, get_image_id, insert_to_milvus

def process_amazon_folder(image_folder, image_embedder, collection):
    image_paths = glob(os.path.join(image_folder, "*.jpg"))
    for img_path in tqdm(image_paths, desc="Processing Amazon images"):
        try:
            img = Image.open(img_path).convert("RGB")
            image_emb = image_embedder.encode_image(img)
            if image_emb is None:
                print(f"Skipping {img_path}, embedding failed.")
                continue

            text = "Amazon product image"  # o usa extract_caption(img) se vuoi descrizioni
            text_emb = get_embedding(text)

            with open(img_path, "rb") as f:
                image_bytes = f.read()
            image_id = get_image_id(image_bytes)

            insert_to_milvus(collection, text_emb, image_emb, text, image_id)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
