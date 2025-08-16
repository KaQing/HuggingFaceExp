#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 22:45:24 2025

@author: user
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# -------------------------------
# 1. Load CSV data
# -------------------------------
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

# Ensure label column is mapped to 0,1,2 (negative, neutral, positive)
label_encoder = LabelEncoder()
train_df['label_enc'] = label_encoder.fit_transform(train_df['label'])
val_df['label_enc'] = label_encoder.transform(val_df['label'])
test_df['label_enc'] = label_encoder.transform(test_df['label'])
num_labels = len(label_encoder.classes_)

# -------------------------------
# 2. Dataset class
# -------------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item

# -------------------------------
# 3. Tokenizer and model
# -------------------------------
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# -------------------------------
# 4. DataLoaders
# -------------------------------
train_dataset = SentimentDataset(train_df['text'].tolist(), train_df['label_enc'].tolist(), tokenizer)
val_dataset = SentimentDataset(val_df['text'].tolist(), val_df['label_enc'].tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# -------------------------------
# 5. Optimizer
# -------------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)

# -------------------------------
# 6. Training loop (1-2 epochs for demo)
# -------------------------------
epochs = 2
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    print(f"Validation Accuracy: {correct/total:.4f}")

# -------------------------------
# 7. Save model
# -------------------------------
model.save_pretrained("multilingual_sentiment_model")
tokenizer.save_pretrained("multilingual_sentiment_model")
print("Model saved to ./multilingual_sentiment_model")
