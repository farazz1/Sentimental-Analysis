import tkinter as tk
from tkinter import messagebox
import joblib

# Load the trained model
model = joblib.load('model.pkl')

def predict():
    review = text_entry.get("1.0", "end").strip()
    if not review:
        messagebox.showwarning("Input Error", "Please enter a review.")
        return
    prediction = model.predict([review])[0]
    result_label.config(text=f"Sentiment: {prediction}")

# Setup GUI window
root = tk.Tk()
root.title("Sentiment Analysis")

tk.Label(root, text="Enter Review Text:").pack(pady=5)
text_entry = tk.Text(root, height=10, width=50)
text_entry.pack(pady=5)

predict_button = tk.Button(root, text="Predict Sentiment", command=predict)
predict_button.pack(pady=5)

result_label = tk.Label(root, text="Sentiment: ", font=("Helvetica", 14))
result_label.pack(pady=10)

root.mainloop()
