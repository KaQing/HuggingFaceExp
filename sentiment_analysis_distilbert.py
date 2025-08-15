#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 19:54:17 2025

@author: user
"""

from transformers import pipeline

# Use a small pre-trained model for sentiment analysis
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",  # small enough for CPU demo
    device=-1  # -1 = CPU
)

# Test it on a sample text
# German sample texts with different emotions
texts = [
    "I love this movie!",
    "This was the worst book I have ever read.",
    "I am so excited about the concert tonight!",
    "I feel really sad about the news.",
    "Wow, I did not see that coming!",
    "I am extremely frustrated with this service.",
    "This chocolate cake is amazing!",
    "I am terrified of spiders.",
    "What a beautiful sunrise this morning.",
    "I am completely overwhelmed by my workload.",
    "This new phone is fantastic!",
    "I hate waiting in long lines.",
    "The vacation was so relaxing and fun!",
    "I feel anxious about the exam tomorrow.",
    "I am grateful for all the help I received.",
    "The traffic today made me so angry!",
    "I am surprised by how quickly the time flew.",
    "I feel very disappointed by the movie ending.",
    "I am thrilled with my promotion at work!",
    "This song makes me feel nostalgic."
    "Ich liebe diesen Film!",
    "Das war das schlechteste Buch, das ich je gelesen habe.",
    "Ich freue mich so auf das Konzert heute Abend!",
    "Ich bin wirklich traurig über die Nachrichten.",
    "Wow, damit habe ich nicht gerechnet!",
    "Ich bin extrem frustriert über diesen Service.",
    "Dieser Schokoladenkuchen ist fantastisch!",
    "Ich habe Angst vor Spinnen.",
    "Was für ein wunderschöner Sonnenaufgang heute Morgen.",
    "Ich bin völlig überwältigt von meiner Arbeitslast.",
    "Dieses neue Handy ist großartig!",
    "Ich hasse es, in langen Schlangen zu warten.",
    "Der Urlaub war so entspannend und schön!",
    "Ich bin nervös wegen der Prüfung morgen.",
    "Ich bin dankbar für die ganze Hilfe, die ich bekommen habe.",
    "Der Verkehr heute hat mich so wütend gemacht!",
    "Ich bin überrascht, wie schnell die Zeit vergangen ist.",
    "Ich bin sehr enttäuscht vom Ende des Films.",
    "Ich freue mich riesig über meine Beförderung!",
    "Dieses Lied macht mich nostalgisch."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Prediction: {result}")
