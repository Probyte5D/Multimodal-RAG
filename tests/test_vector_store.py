import unittest
from pymilvus import utility
from models.vector_store import (
    init_milvus_collection,
    get_embedding,
    get_image_id,
    insert_to_milvus,
    search_similar
)



class TestMilvusImageText(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.collection_name = "test_rag_image_texts"
        cls.collection = init_milvus_collection(name=cls.collection_name)

        cls.test_text = "Una mela rossa su un tavolo"
        cls.test_image_bytes = b"immagine_finta"
        cls.text_emb = get_embedding(cls.test_text)
        cls.image_emb = [0.01] * 512
        cls.image_id = get_image_id(cls.test_image_bytes)

    def test_1_insert_and_deduplication(self):
        # Primo inserimento
        insert_to_milvus(self.collection, self.text_emb, self.image_emb, self.test_text, self.image_id)

        # Secondo inserimento (dovrebbe non inserire)
        insert_to_milvus(self.collection, self.text_emb, self.image_emb, self.test_text, self.image_id)

        # Controlla che esista solo un dato
        expr = f'image_id == "{self.image_id}"'
        results = self.collection.query(expr=expr, output_fields=["text"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], self.test_text)

    def test_2_search_similarity(self):
        # Ricerca simile (escludendo stesso ID -> nessun risultato)
        simili = search_similar(self.collection, self.text_emb, exclude_image_id=self.image_id)
        self.assertEqual(len(simili), 0)

    @classmethod
    def tearDownClass(cls):
        # Elimina la collection dopo il test
        cls.collection.release()
        utility.drop_collection(cls.collection_name)

if __name__ == '__main__':
    unittest.main()
