"""
Machine Learning Pipeline Demo
Dual-Engine Text Classification:
1. Spam vs. Ham Classifier (Multinomial Naive Bayes)
2. Customer Sentiment Analyzer (Logistic Regression)
"""

import os
import sys
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

from src.text_cleaner import TextCleaner


def build_spam_dataset() -> Tuple[List[str], List[str]]:
    """Returns sample dataset for spam vs ham classification."""
    texts = [
        "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim.",
        "URGENT: Your account has been suspended. Call 1800123456 now to verify.",
        "Win cash prizes instantly! Visit www.free-lottery-rewards.com right now.",
        "Special loan offer approved up to $50,000 with 0% interest rate.",
        "Exclusive deal: buy 1 get 2 free watches! Limited stock remaining.",
        "Claim your lottery winnings of $500,000 today by replying with your details.",
        "Hey, are we still meeting for lunch at 1 PM today?",
        "Please find attached the quarterly project review slides.",
        "Can you send me the updated notes from yesterday's machine learning lecture?",
        "Don't forget to push your code changes to GitHub before EOD.",
        "Thanks for helping me debug the model training script earlier.",
        "I will be running late for the meeting, please start without me."
    ]
    labels = [
        "spam", "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham", "ham"
    ]
    return texts, labels


def build_sentiment_dataset() -> Tuple[List[str], List[str]]:
    """Returns sample dataset for sentiment analysis."""
    texts = [
        "This product is amazing and exceeded all my expectations! Highly recommend.",
        "Outstanding customer support, very quick and helpful response.",
        "The model works beautifully with high precision and low latency.",
        "Great quality build, looks very premium and durable.",
        "Terrible experience, arrived broken and support was unhelpful.",
        "Poor quality material and completely useless instructions.",
        "Very disappointed with the performance, constantly crashes.",
        "Waste of money, do not buy this product."
    ]
    labels = [
        "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative"
    ]
    return texts, labels


def train_spam_classifier():
    cleaner = TextCleaner(remove_stopwords=True)
    raw_texts, labels = build_spam_dataset()
    cleaned_texts = cleaner.batch_clean(raw_texts)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])
    pipeline.fit(cleaned_texts, labels)
    return cleaner, pipeline


def train_sentiment_classifier():
    cleaner = TextCleaner(remove_stopwords=True, preserve_negations=True)
    raw_texts, labels = build_sentiment_dataset()
    cleaned_texts = cleaner.batch_clean(raw_texts)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(random_state=42))
    ])
    pipeline.fit(cleaned_texts, labels)
    return cleaner, pipeline


def main():
    print("=" * 60)
    print("SmartNLP: Dual-Engine Text Classification System")
    print("=" * 60)

    # 1. Spam Classification Demo
    print("\n--- Training Spam Classifier ---")
    spam_cleaner, spam_model = train_spam_classifier()
    print("Spam classifier trained successfully.")

    sample_messages = [
        "Hey friend, let's catch up this weekend over coffee.",
        "WINNER! Call 9876543210 to claim your $5,000 cash bonus now!"
    ]
    for msg in sample_messages:
        cleaned = spam_cleaner.clean(msg)
        pred = spam_model.predict([cleaned])[0]
        probs = spam_model.predict_proba([cleaned])[0]
        classes = spam_model.classes_
        prob_dict = dict(zip(classes, [round(p, 4) for p in probs]))
        print(f"\nMessage: '{msg}'")
        print(f"Prediction: [{pred.upper()}] (Probabilities: {prob_dict})")

    # 2. Sentiment Analysis Demo
    print("\n" + "=" * 60)
    print("--- Training Customer Sentiment Analyzer ---")
    sent_cleaner, sent_model = train_sentiment_classifier()
    print("Sentiment analyzer trained successfully.")

    sample_reviews = [
        "The software update made everything so smooth and fast, absolutely love it!",
        "Total waste of time, failed to load any data and crashed repeatedly."
    ]
    for review in sample_reviews:
        cleaned = sent_cleaner.clean(review)
        pred = sent_model.predict([cleaned])[0]
        probs = sent_model.predict_proba([cleaned])[0]
        classes = sent_model.classes_
        prob_dict = dict(zip(classes, [round(p, 4) for p in probs]))
        print(f"\nReview: '{review}'")
        print(f"Sentiment: [{pred.upper()}] (Probabilities: {prob_dict})")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
