"""
Word Analytics Module
Count word occurrences in chat, filtered by participant
"""

import pandas as pd
from typing import Dict, Optional
import re


def count_word(df: pd.DataFrame, word: str, participant: Optional[str] = None) -> Dict:
    """
    Count occurrences of a word in the chat.
    
    Args:
        df: DataFrame with 'sender' and 'message' columns
        word: The word to search for (case-insensitive)
        participant: Optional - filter by specific participant, None for all
    
    Returns:
        Dict with 'total' count and 'by_participant' breakdown
    """
    
    if df.empty or 'message' not in df.columns:
        return {'total': 0, 'by_participant': {}}
    
    # Filter by participant if specified
    if participant and participant != 'All':
        filtered_df = df[df['sender'] == participant].copy()
    else:
        filtered_df = df.copy()
    
    if filtered_df.empty:
        return {'total': 0, 'by_participant': {}}
    
    # Create regex pattern for word matching (case-insensitive, word boundaries)
    # Using word boundaries to match whole words, but also allowing partial matches
    pattern = re.compile(re.escape(word), re.IGNORECASE)
    
    # Count occurrences in each message
    filtered_df['count'] = filtered_df['message'].apply(
        lambda x: len(pattern.findall(str(x))) if pd.notna(x) else 0
    )
    
    # Calculate total
    total = filtered_df['count'].sum()
    
    # Group by participant
    by_participant = filtered_df.groupby('sender')['count'].sum().to_dict()
    
    # Sort by count descending
    by_participant = dict(sorted(by_participant.items(), key=lambda x: x[1], reverse=True))
    
    # Filter out zero counts
    by_participant = {k: v for k, v in by_participant.items() if v > 0}
    
    return {
        'total': int(total),
        'by_participant': by_participant
    }


def get_top_words(df: pd.DataFrame, participant: Optional[str] = None, top_n: int = 20) -> Dict[str, int]:
    """
    Get the most frequently used words by a participant or in the whole chat.
    
    Args:
        df: DataFrame with 'sender' and 'message' columns
        participant: Optional - filter by specific participant
        top_n: Number of top words to return
    
    Returns:
        Dict of word -> count
    """
    
    if df.empty or 'message' not in df.columns:
        return {}
    
    # Filter by participant if specified
    if participant and participant != 'All':
        filtered_df = df[df['sender'] == participant]
    else:
        filtered_df = df
    
    # Combine all messages
    all_text = ' '.join(filtered_df['message'].dropna().astype(str))
    
    # Tokenize (simple word extraction)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
        'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'they',
        'this', 'that', 'with', 'from', 'will', 'would', 'there', 'their', 'what',
        'about', 'which', 'when', 'make', 'like', 'just', 'know', 'take', 'into',
        'year', 'your', 'some', 'could', 'them', 'than', 'then', 'now', 'look',
        'only', 'come', 'its', 'also', 'back', 'after', 'use', 'how', 'man',
        'media', 'omitted', 'deleted', 'message'
    }
    
    # Count words
    word_counts = {}
    for word in words:
        if word not in stop_words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort and get top N
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    return dict(sorted_words[:top_n])

