#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 20:46:11 2025

@author: user
"""

from datasets import load_dataset
import pandas as pd

train_sample_size = 5000
val_sample_size = 1000
test_sample_size = 1000

# Load the dataset
ds = load_dataset("clapAI/MultiLingualSentiment")

# Sample from the train split
train_sample_ds = ds['train'].shuffle(seed=42).select(range(train_sample_size))
train_df = pd.DataFrame(train_sample_ds)[["text", "label", "language"]]
train_df.to_csv("train.csv", index=False)
print("Train sample saved to train_multilingual_sentiment.csv")
print(train_df.head())

# Sample from the validation split
val_sample_ds = ds['validation'].shuffle(seed=42).select(range(val_sample_size))
val_df = pd.DataFrame(val_sample_ds)[["text", "label", "language"]]
val_df.to_csv("val.csv", index=False)
print("Validation sample saved to val_multilingual_sentiment.csv")
print(val_df.head())

# Sample from the test split
test_sample_ds = ds['test'].shuffle(seed=42).select(range(test_sample_size))
test_df = pd.DataFrame(test_sample_ds)[["text", "label", "language"]]
test_df.to_csv("test.csv", index=False)
print("Test sample saved to test_multilingual_sentiment.csv")
print(test_df.head())