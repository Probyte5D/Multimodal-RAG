import os
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import hashlib
import numpy as np

host = os.getenv("MILVUS_HOST", "localhost")
port = os.getenv("MILVUS_PORT", "19530")


# Connessione Milvus + setup collection
def init_milvus_collection(name="rag_image_texts"):
    connections.connect("default", host=host, port=port)

    # Ora creiamo la collection da zero con lo schema aggiornato
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields, description="Image text embeddings with image ID")
    collection = Collection(name, schema)

    # Crea indice su entrambi gli embedding
    collection.create_index("text_embedding", {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    })
    collection.create_index("image_embedding", {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    })

    collection.load()
    return collection

# Embedding testo con SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return embedder.encode([text])[0].tolist()

def get_image_id(image_bytes: bytes):
    return hashlib.md5(image_bytes).hexdigest()

def exists_text_with_image(collection, text, image_id):
    expr = f'image_id == "{image_id}"'
    try:
        results = collection.query(expr=expr, output_fields=["text"])
        return any(r["text"] == text for r in results)
    except Exception as e:
        print(f"Query failed: {e}")
        return False

def insert_to_milvus(collection, text_embedding, image_embedding, text, image_id, partition_name=None):
    expr = f'image_id == "{image_id}"'
    try:
        results = collection.query(expr=expr, output_fields=["text"])
        exists = any(r["text"] == text for r in results)
        if exists:
            print(f"⚠️ L'immagine con image_id={image_id} e testo già esiste.")
            return
    except Exception as e:
        print(f"Query failed: {e}")

    try:
        data = [
            [image_id],
            [text],
            [text_embedding],
            [image_embedding],
        ]
        collection.insert(data, partition_name=partition_name)
        print(f"✅ Inserita image_id={image_id} nella partizione '{partition_name}'")
    except Exception as e:
        print(f"❌ Errore durante l'inserimento di image_id={image_id}: {e}")


def search_similar(collection, query_embedding, anns_field="text_embedding", exclude_image_id=None, top_k=5, threshold=0.6):
    expr = f'image_id != "{exclude_image_id}"' if exclude_image_id else None
    results = collection.search(
        data=[query_embedding],
        anns_field=anns_field,
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        expr=expr,
        output_fields=["text", "image_id"]
    )
    filtered = []
    for hit in results[0]:
        if hit.distance < threshold:
            filtered.append((hit.entity.get("text"), hit.entity.get("image_id")))
    return filtered


# 🔍 NUOVA FUNZIONE: Ricerca semantica per testo (non modifica le precedenti)
def search_similar_by_text(collection, query_text: str, top_k=5, threshold=0.6):
    query_embedding = get_embedding(query_text)
    results = collection.search(
        data=[query_embedding],
        anns_field="text_embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text", "image_id"]
    )
    filtered = []
    for hit in results[0]:
        if hit.distance < threshold:
            filtered.append({
                "caption": hit.entity.get("text"),
                "image_id": hit.entity.get("image_id"),
                "distance": hit.distance
            })
    return filtered
