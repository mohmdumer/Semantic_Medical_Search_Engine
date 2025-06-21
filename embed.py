# semantic_search_medical/embed.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def load_cleaned_data(csv_path=r'E:\Portfolio\Semantic_Search_Engine\data\clean_medquad.csv'):
    df = pd.read_csv(csv_path)
    assert 'question' in df.columns and 'answer' in df.columns
    return df

def generate_embeddings(df, model_name='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb', batch_size=32):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    questions = df['question'].tolist()
    print(f"Generating embeddings for {len(questions)} questions...")

    embeddings = model.encode(questions, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings)

def save_embeddings(embeddings, path='data/question_embeddings.npy'):
    np.save(path, embeddings)
    print(f"Saved embeddings to {path}")

if __name__ == '__main__':
    df = load_cleaned_data()
    embeddings = generate_embeddings(df)
    save_embeddings(embeddings)
