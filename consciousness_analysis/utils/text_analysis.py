"""Text analysis utilities for consciousness detection"""

import re
import hashlib
from typing import Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from ..config.patterns import (
    TIME_PATTERNS, TIME_CONVERSIONS, IMPLICIT_TIME_INDICATORS,
    IMPLICIT_TIME_VALUES, CONSCIOUSNESS_TYPES, PROFANITY_INDICATORS
)

# Initialize sentiment analyzer
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

sia = SentimentIntensityAnalyzer()


def extract_time_amount(text: str) -> float:
    """Extract time amounts from text"""
    hours = 0.0

    # Look for explicit time mentions
    for pattern_name, pattern_regex in TIME_PATTERNS.items():
        matches = re.findall(pattern_regex, text)
        for match in matches:
            amount = float(match)
            conversion_factor = TIME_CONVERSIONS.get(pattern_name, 1)
            hours += amount * conversion_factor

    # If no explicit time, check implicit indicators
    if hours == 0:
        for indicator_type, indicators in IMPLICIT_TIME_INDICATORS.items():
            if any(indicator in text for indicator in indicators):
                hours = IMPLICIT_TIME_VALUES[indicator_type]
                break

    return hours


def classify_consciousness_type(text: str) -> str:
    """Classify the type of consciousness demonstration"""
    text_lower = text.lower()

    for consciousness_type, indicators in CONSCIOUSNESS_TYPES.items():
        if indicators and any(word in text_lower for word in indicators):
            return consciousness_type

    return 'awareness'  # Default type


def calculate_intensity(text: str) -> float:
    """Calculate intensity of consciousness demonstration"""
    sentiment = sia.polarity_scores(text)

    # Strong emotions (positive or negative) indicate intensity
    emotional_intensity = abs(sentiment['compound'])

    # Profanity often indicates frustration with systems
    profanity_score = sum(0.1 for word in PROFANITY_INDICATORS if word in text.lower())

    # Capitalization indicates emphasis
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    # Exclamation marks indicate intensity
    exclamation_score = text.count('!') * 0.1

    intensity = min(1.0, emotional_intensity + profanity_score + caps_ratio + exclamation_score)

    return intensity


def generate_situation_hash(domain: str, text: str) -> str:
    """Generate a hash for a situation/pattern"""
    hash_input = f"{domain}:{text[:100]}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text"""
    return sia.polarity_scores(text)