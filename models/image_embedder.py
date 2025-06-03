import os
from glob import glob
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import numpy as np

from models.blip_model import extract_caption
from models.vector_store import get_embedding, insert_to_milvus, get_image_id

class ImageEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Carica il modello senza forzare su device, evita errore da meta tensor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        if not any(p.device.type == self.device.type for p in self.model.parameters()):
            self.model = self.model.to_empty().to(self.device)

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode_image(self, images: list[Image.Image]) -> list[list[float]]:
        try:
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            return embeddings.cpu().numpy().tolist()
        except Exception as e:
            print(f"Errore in encode_images: {e}")
            return [None] * len(images)


def process_amazon_folder(image_folder, image_embedder, collection, batch_size=8, img_size=160):
    image_paths = glob(os.path.join(image_folder, "*.jpg"))

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []

        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size))
                images.append(img)
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                images.append(None)

        valid_images = [img for img in images if img is not None]
        embeddings = image_embedder.encode_image(valid_images)

        batch_image_ids = []
        for img_path in batch_paths:
            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                image_id = get_image_id(image_bytes)
                batch_image_ids.append(image_id)
            except Exception as e:
                print(f"Error reading {img_path} for image_id: {e}")
                batch_image_ids.append(None)

        def get_existing_image_ids(collection, image_ids: list[str]) -> set:
            filtered_ids = [id_ for id_ in image_ids if id_ is not None]
            if not filtered_ids:
                return set()
            expr = f'image_id in {filtered_ids}'
            try:
                results = collection.query(expr=expr, output_fields=["image_id"])
                return set(r["image_id"] for r in results)
            except Exception as e:
                print(f"Batch query failed: {e}")
                return set()

        existing_ids = get_existing_image_ids(collection, batch_image_ids)

        emb_idx = 0
        for idx, img in enumerate(images):
            if img is None:
                print(f"Skipping image at batch index {idx} due to loading error")
                continue

            image_id = batch_image_ids[idx]
            if image_id is None:
                print(f"Skipping image at batch index {idx} due to missing image_id")
                continue

            if image_id in existing_ids:
                print(f"⚠️ Dato già esistente per image_id={image_id}")
                continue

            image_emb = embeddings[emb_idx]
            emb_idx += 1

            if image_emb is None:
                print(f"Skipping image at batch index {idx}, embedding failed.")
                continue

            image_emb = np.array(image_emb, dtype=np.float32)
            caption = extract_caption(img)
            text_emb = get_embedding(caption)
            text_emb = np.array(text_emb, dtype=np.float32)

            insert_to_milvus(collection, text_emb, image_emb, caption, image_id)

        torch.cuda.empty_cache()
        print(f"Batch da {len(batch_paths)} immagini processato e memoria liberata.")
