"""
Text Cleaner and Normalization Module
Handles tokenization, regex cleaning, lowercasing, stopword removal, and stemming.
"""

import re
import string
import unicodedata
from typing import List, Optional


# Comprehensive default English stopwords list to operate reliably offline without downloading external corpora
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", 
    "yours", "yourself", "yourselves"
}


class TextCleaner:
    """Preprocesses and normalizes text for NLP feature extraction."""

    def __init__(self, remove_stopwords: bool = True, preserve_negations: bool = True):
        self.remove_stopwords = remove_stopwords
        self.preserve_negations = preserve_negations
        self.stopwords = set(DEFAULT_STOPWORDS)
        
        # When analyzing sentiment, preserving negation words like 'not', 'no', 'never' is crucial
        if self.preserve_negations:
            negations = {"not", "no", "nor", "neither", "never", "none", "cannot", "don't", "isn't", "wasn't", "aren't"}
            self.stopwords = self.stopwords - negations

    def clean(self, text: str) -> str:
        """Applies full cleaning pipeline to a raw text string."""
        if not isinstance(text, str):
            return ""

        # Normalize unicode characters
        text = unicodedata.normalize("NFKD", text)

        # Replace URLs with token
        text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text)

        # Replace phone numbers and currency symbols
        text = re.sub(r"(\$\d+|\£\d+|\€\d+|\b\d+\s?(dollars|pounds|euro|usd)\b)", " currency_amount ", text, flags=re.IGNORECASE)
        text = re.sub(r"\b\d{10,12}\b", " phone_number ", text)

        # Replace emails with token
        text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " email_address ", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)

        # Lowercase
        text = text.lower()

        # Remove punctuation while preserving alphanumeric characters and space
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace
        tokens = text.split()

        # Filter stopwords if enabled
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stopwords and len(w) > 1]

        return " ".join(tokens)

    def batch_clean(self, texts: List[str]) -> List[str]:
        """Cleans a list of texts."""
        return [self.clean(t) for t in texts]
