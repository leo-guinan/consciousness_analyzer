"""Filters for detecting and excluding memes, copypasta, and viral content"""

import re
from typing import Dict, List
import hashlib

# Known meme/copypasta patterns
KNOWN_MEMES = [
    "william: doctor, i need help",
    "this is just to say",
    "i have eaten the plums",
    "forgive me they were delicious",
    "so sweet and so cold",
    # Add more known memes as discovered
]

# Patterns that indicate viral/meme content
MEME_INDICATORS = [
    r"RT @\w+:",  # Retweets
    r"^@\w+ @\w+ @\w+",  # Multiple mentions (likely conversations)
    r"(?i)(copypasta|meme|viral|joke)",
    r"(?i)(lmao|lmfao|rofl|😂{3,})",  # Excessive laughter
    r"(?i)^(fellas|guys|ladies|kings|queens) is it",  # Common meme formats
]

def is_likely_meme(text: str) -> bool:
    """
    Check if text is likely a meme or copypasta

    Args:
        text: Tweet text to check

    Returns:
        True if text appears to be a meme/copypasta
    """
    text_lower = text.lower()

    # Check for known memes
    for meme in KNOWN_MEMES:
        if meme.lower() in text_lower:
            return True

    # Check for meme patterns
    for pattern in MEME_INDICATORS:
        if re.search(pattern, text):
            return True

    # Check for excessive repetition of emojis
    emoji_pattern = r'[\U0001F300-\U0001F9FF]'
    emojis = re.findall(emoji_pattern, text)
    if len(emojis) > 5:  # Too many emojis usually indicates non-serious content
        return True

    return False

def detect_duplicate_content(tweets: List[Dict]) -> Dict[str, List[str]]:
    """
    Detect tweets with identical or near-identical content across multiple users

    Args:
        tweets: List of tweet dictionaries with 'content' and 'user' fields

    Returns:
        Dictionary mapping content hashes to list of users who posted it
    """
    content_map = {}

    for tweet in tweets:
        # Normalize content for comparison
        content = tweet.get('content', '').lower().strip()
        # Remove mentions and URLs for comparison
        content_normalized = re.sub(r'@\w+', '', content)
        content_normalized = re.sub(r'https?://\S+', '', content_normalized)
        content_normalized = re.sub(r'\s+', ' ', content_normalized).strip()

        # Create hash of normalized content
        if len(content_normalized) > 50:  # Only check substantial content
            content_hash = hashlib.md5(content_normalized.encode()).hexdigest()

            if content_hash not in content_map:
                content_map[content_hash] = {
                    'content': content[:200],
                    'users': []
                }

            user = tweet.get('user', tweet.get('email', 'unknown'))
            if user not in content_map[content_hash]['users']:
                content_map[content_hash]['users'].append(user)

    # Return only content that appears across multiple users
    duplicates = {
        hash_val: data
        for hash_val, data in content_map.items()
        if len(data['users']) > 1
    }

    return duplicates

def is_genuine_consciousness_indicator(text: str, domain: str = None) -> bool:
    """
    Validate if text represents a genuine consciousness indicator

    Args:
        text: Tweet text to validate
        domain: The domain being analyzed (healthcare, financial, etc.)

    Returns:
        True if text appears to be genuine consciousness indicator
    """
    # First check if it's a meme
    if is_likely_meme(text):
        return False

    # Check for personal pronouns indicating first-person experience
    personal_indicators = [
        r'\b(I|my|me|I\'ve|I\'m|I\'ll)\b',
        r'\b(we|our|us|we\'ve|we\'re)\b'
    ]

    has_personal = any(re.search(pattern, text, re.IGNORECASE) for pattern in personal_indicators)

    # Check for specific time indicators suggesting real experience
    time_indicators = [
        r'\b(today|yesterday|last week|last month|this morning)\b',
        r'\b(finally|after \d+|spent \d+|took \d+)\b',
        r'\b(\d+ hours?|\d+ days?|\d+ months?)\b'
    ]

    has_time = any(re.search(pattern, text, re.IGNORECASE) for pattern in time_indicators)

    # Check for action verbs indicating real activity
    action_indicators = [
        r'\b(called|emailed|went to|submitted|applied|fought|argued|explained)\b',
        r'\b(waiting|trying|dealing|struggling|fighting)\b'
    ]

    has_action = any(re.search(pattern, text, re.IGNORECASE) for pattern in action_indicators)

    # Require at least personal experience + one other indicator
    if has_personal and (has_time or has_action):
        return True

    # Check for detailed specificity that indicates real experience
    if domain:
        # Domain-specific validation
        if domain == 'healthcare':
            specific_patterns = [
                r'\b(claim #?\d+|case #?\d+|reference #?\d+)\b',
                r'\b(prior auth|PA request|appeal \d+|tier \d+)\b',
                r'\b(copay|deductible|out of pocket|EOB)\b'
            ]
        elif domain == 'financial':
            specific_patterns = [
                r'\b(account ending in \d+|case #?\d+|ticket #?\d+)\b',
                r'\b(APR|interest rate|minimum payment|balance transfer)\b',
                r'\b(chapter \d+|collection agency|credit score \d+)\b'
            ]
        else:
            specific_patterns = []

        has_specific = any(re.search(pattern, text, re.IGNORECASE) for pattern in specific_patterns)
        if has_specific:
            return True

    # If still uncertain, check length and complexity
    # Real experiences tend to be more detailed
    if len(text) > 100 and has_personal:
        return True

    return False

def filter_consciousness_indicators(indicators: List[Dict]) -> List[Dict]:
    """
    Filter a list of consciousness indicators to remove memes and duplicates

    Args:
        indicators: List of indicator dictionaries

    Returns:
        Filtered list of genuine indicators
    """
    # First, detect duplicates
    duplicates = detect_duplicate_content(indicators)
    duplicate_contents = set()

    for dup_data in duplicates.values():
        if len(dup_data['users']) > 2:  # If 3+ users have same content, likely meme
            duplicate_contents.add(dup_data['content'][:100])

    # Filter indicators
    filtered = []
    for indicator in indicators:
        content = indicator.get('content', '')
        domain = indicator.get('complexity_domain', '')

        # Skip if it's a known duplicate
        if any(dup in content for dup in duplicate_contents):
            continue

        # Skip if it's not genuine
        if not is_genuine_consciousness_indicator(content, domain):
            continue

        filtered.append(indicator)

    return filtered