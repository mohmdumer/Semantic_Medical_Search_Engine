# semantic_search_medical/search.py

import numpy as np
from sentence_transformers import SentenceTransformer
from index import FaissIndexer

class SemanticMedicalSearcher:
    def __init__(self,
                 model_name: str = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
                 index_path: str = 'models/faiss_index.index',
                 dim: int = 768):

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.indexer = FaissIndexer(dim=dim, index_path=index_path)
        self.indexer.load()

    def embed_query(self, query: str):
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding.astype('float32')

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.embed_query(query)
        results = self.indexer.search(query_embedding, top_k=top_k)
        return results

if __name__ == '__main__':
    engine = SemanticMedicalSearcher()
    print("🔎 Medical Semantic Search Engine Ready")
    while True:
        query = input("\nEnter your medical question (or 'exit'): ")
        if query.lower() == 'exit':
            break
        top_matches = engine.search(query)
        print("\nTop Results:\n")
        for i, (text, score) in enumerate(top_matches):
            print(f"Result #{i+1} (Score: {score:.4f}):\n{text}\n{'-'*60}")
