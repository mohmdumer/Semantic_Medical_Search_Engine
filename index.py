# semantic_search_medical/index.py

import faiss
import numpy as np
import pandas as pd
import os

class FaissIndexer:
    def __init__(self, dim: int, index_path: str = 'models/faiss_index.index'):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dim)  # Inner product since embeddings are normalized
        self.text_data = []

    def build(self, embeddings: np.ndarray, text_data: list):
        assert embeddings.shape[1] == self.dim
        print("Building FAISS index...")
        self.index.add(embeddings)
        self.text_data = text_data

    def save(self):
        print(f"Saving FAISS index to {self.index_path}")
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path.replace('.index', '_texts.npy'), 'wb') as f:
            np.save(f, np.array(self.text_data))

    def load(self):
        print(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        self.text_data = np.load(self.index_path.replace('.index', '_texts.npy'), allow_pickle=True).tolist()

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding.astype('float32'), top_k)
        results = [(self.text_data[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results

if __name__ == '__main__':
    # Load data & embeddings
    df = pd.read_csv('data/clean_medquad.csv')
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    combined = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]

    embeddings = np.load('data/question_embeddings.npy')

    # Build FAISS index
    indexer = FaissIndexer(dim=embeddings.shape[1])
    indexer.build(embeddings, combined)
    os.makedirs(os.path.dirname(indexer.index_path), exist_ok=True)
    indexer.save()
