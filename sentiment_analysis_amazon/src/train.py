import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from collections import Counter
import joblib

def load_data(path):
    df = pd.read_csv(path)
    df = df[['Text', 'Score']].dropna()

    def label_sentiment(score):
        if score >= 4:
            return 'Positive'
        elif score <= 2:
            return 'Negative'
        else:
            return None  # Skip neutral scores

    df['Sentiment'] = df['Score'].apply(label_sentiment)
    df = df[df['Sentiment'].notnull()]

    # Balance the dataset by undersampling the majority class
    pos = df[df['Sentiment'] == 'Positive']
    neg = df[df['Sentiment'] == 'Negative']
    min_len = min(len(pos), len(neg))

    balanced_df = pd.concat([pos.sample(min_len, random_state=42), neg.sample(min_len, random_state=42)])

    print("Label distribution after balancing:", Counter(balanced_df['Sentiment']))

    return balanced_df['Text'].tolist(), balanced_df['Sentiment'].tolist()

def train_model():
    texts, labels = load_data('../Data/train_40k.csv')

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    model = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('classifier', MultinomialNB()),
    ])

    print(f"Training on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Accuracy on test set: {accuracy_score(y_test, preds):.2f}")

    joblib.dump(model, 'model.pkl')
    print("Model saved to 'model.pkl'")

if __name__ == "__main__":
    train_model()
