import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

def load_data():
    df = pd.read_csv(
        "/Users/justmedevesh/Desktop/Softwarica/Information_retrevial/Assignment/Document_Classification/data/documents.csv",
        encoding="latin-1"
    )

    # 🔑 FIX: normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Now this will ALWAYS work
    texts = df["text"].apply(preprocess).tolist()
    labels = df["category"].tolist()

    return texts, labels