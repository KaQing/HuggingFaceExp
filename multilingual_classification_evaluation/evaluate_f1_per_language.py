#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 22:59:13 2025

@author: user
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

# -------------------------------
# 1. Load the CSV
# -------------------------------
df = pd.read_csv("test_predicted.csv")  # must have 'label', 'predicted_label', 'language'

# -------------------------------
# 2. Encode labels (3 classes)
# -------------------------------
classes = ["negative", "neutral", "positive"]
le = LabelEncoder()
le.fit(classes)
df['label_enc'] = le.transform(df['label'])
df['predicted_label_enc'] = le.transform(df['predicted_label'])

# -------------------------------
# 3. Compute F1 score per language
# -------------------------------
languages = df['language'].unique()
for lang in languages:
    df_lang = df[df['language'] == lang]
    f1 = f1_score(df_lang['label_enc'], df_lang['predicted_label_enc'], average='weighted')
    print(f"Language: {lang} | F1 score (weighted): {f1:.4f}")
    
    """
    # Pass explicit labels to avoid mismatch
    print(classification_report(
        df_lang['label_enc'],
        df_lang['predicted_label_enc'],
        labels=[0,1,2],          # always use 3 classes
        target_names=classes,
        zero_division=0))           # avoids errors if a class is missing
    """
    
