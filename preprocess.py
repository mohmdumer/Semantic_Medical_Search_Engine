import pandas as pd
import re

def clean_text(text: str) -> str:
    """Normalize and clean medical Q&A text."""
    text = re.sub(r'\s+', ' ', str(text))  # Replace multiple spaces with single space
    text = re.sub(r'[^\w\s,?.!-]', '', text)  # Remove unwanted characters
    return text.strip()

def load_medquad_csv(csv_path: str = r'E:\Portfolio\Semantic_Search_Engine\data\medquad.csv') -> pd.DataFrame:
    """Load MedQuAD from CSV and clean it."""
    df = pd.read_csv(csv_path)
    print("Columns in CSV:", df.columns.tolist())

    # Rename to standard column names if needed
    if 'question' not in df.columns or 'answer' not in df.columns:
        possible_q = [c for c in df.columns if 'question' in c.lower()]
        possible_a = [c for c in df.columns if 'answer' in c.lower()]
        assert possible_q and possible_a, "CSV must include question/answer columns"
        df = df.rename(columns={possible_q[0]: 'question', possible_a[0]: 'answer'})

    df = df[['question', 'answer']].dropna()
    df['question'] = df['question'].apply(clean_text)
    df['answer'] = df['answer'].apply(clean_text)
    df = df[df['question'].str.len() > 10]

    print(f"✅ Loaded {len(df)} valid Q&A pairs.")
    return df

if __name__ == '__main__':
    df = load_medquad_csv()
    df.to_csv(r'E:\Portfolio\Semantic_Search_Engine\data\clean_medquad.csv', index=False)
    print("✅ Cleaned CSV saved successfully.")
    print(df.head())
    print("✅ Preprocessing complete.")