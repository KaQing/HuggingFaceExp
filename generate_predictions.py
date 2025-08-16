#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 22:53:29 2025

@author: user
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# 1. Load model and tokenizer
# -------------------------------
model_path = "./multilingual_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)
model.eval()

# -------------------------------
# 2. Load test CSV
# -------------------------------
test_df = pd.read_csv("test.csv")  # must have 'text' and 'label' columns

# -------------------------------
# 3. Prediction function
# -------------------------------
def predict_sentiment(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            results.extend(preds)
    return results

# -------------------------------
# 4. Encode labels
# -------------------------------
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test_df['label_enc'] = le.fit_transform(test_df['label'])  # map labels to 0/1/2

# -------------------------------
# 5. Predict
# -------------------------------
test_preds = predict_sentiment(test_df['text'].tolist())
test_df['predicted_label_enc'] = test_preds
test_df['predicted_label'] = le.inverse_transform(test_preds)

# -------------------------------
# 6. Compute accuracy
# -------------------------------
accuracy = (test_df['label_enc'] == test_df['predicted_label_enc']).mean()
print(f"Test set accuracy: {accuracy:.4f}")

# -------------------------------
# 7. Save predictions
# -------------------------------
test_df.to_csv("test_predicted.csv", index=False)
print("Predictions saved to test_predicted.csv")
print(test_df.head())
