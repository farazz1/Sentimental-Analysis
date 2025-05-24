import pandas as pd
import re
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['Text', 'Score']].dropna()

    def get_label(score):
        if score >= 4:
            return 1  # Positive
        elif score <= 2:
            return 0  # Negative
        else:
            return None

    df['label'] = df['Score'].apply(get_label)
    df = df[df['label'].notnull()]
    return df['Text'].tolist(), df['label'].astype(int).tolist()

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text)  # remove multiple spaces
    return text.strip().lower()

def save_model(model, vectorizer, filepath='model.pkl'):
    joblib.dump((model, vectorizer), filepath)

def load_model(filepath='model.pkl'):
    return joblib.load(filepath)
