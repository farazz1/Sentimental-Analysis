import joblib

def predict_sentiment(text):
    model = joblib.load('model.pkl')
    prediction = model.predict([text])[0]
    return prediction

if __name__ == "__main__":
    while True:
        user_input = input("Enter a review (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        sentiment = predict_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}")
