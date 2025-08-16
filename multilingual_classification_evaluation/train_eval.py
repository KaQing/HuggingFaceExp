#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 22:10:26 2025

@author: user
"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("train_multilingual_sentiment.csv")  # Columns: text, label, language

# Map labels to integers
label_map = {"positive": 0, "neutral": 1, "negative": 2}
df["label"] = df["label"].map(label_map)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df[["text", "label"]])  # Ignore language for now



train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": accuracy, "f1": f1}

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

from transformers import pipeline

classifier = pipeline("text-classification", model="./trained_model", tokenizer="./trained_model")
text = "This is a sample text to classify."
prediction = classifier(text)
print(prediction)  # Returns label and score