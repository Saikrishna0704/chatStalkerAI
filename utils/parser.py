"""
WhatsApp Chat Parser
Parses exported WhatsApp chat .txt files into structured data
"""

import re
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional


def parse_whatsapp_chat(file_content: str) -> pd.DataFrame:
    """
    Parse WhatsApp exported chat text into a DataFrame.
    
    Handles multiple date formats:
    - DD/MM/YYYY, HH:MM - Sender: Message
    - MM/DD/YY, HH:MM AM/PM - Sender: Message
    - [DD/MM/YYYY, HH:MM:SS] Sender: Message
    
    Returns DataFrame with columns: timestamp, sender, message
    """
    
    # Common WhatsApp export patterns
    patterns = [
        # Pattern 1: DD/MM/YYYY, HH:MM - Sender: Message
        r'(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\s*-\s*([^:]+):\s*(.*)',
        # Pattern 2: [DD/MM/YYYY, HH:MM:SS] Sender: Message
        r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]\s*([^:]+):\s*(.*)',
    ]
    
    messages = []
    lines = file_content.split('\n')
    
    current_message = None
    
    for line in lines:
        matched = False
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                # Save previous message if exists
                if current_message:
                    messages.append(current_message)
                
                timestamp_str, sender, message = match.groups()
                
                # Parse timestamp
                timestamp = parse_timestamp(timestamp_str)
                
                # Clean sender name
                sender = sender.strip()
                
                # Skip system messages
                if is_system_message(sender, message):
                    current_message = None
                    matched = True
                    break
                
                current_message = {
                    'timestamp': timestamp,
                    'sender': sender,
                    'message': message.strip()
                }
                matched = True
                break
        
        # If no pattern matched, it might be a continuation of previous message
        if not matched and current_message and line.strip():
            current_message['message'] += '\n' + line.strip()
    
    # Don't forget the last message
    if current_message:
        messages.append(current_message)
    
    df = pd.DataFrame(messages)
    
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'sender', 'message'])
    
    return df


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse various timestamp formats from WhatsApp exports."""
    
    formats = [
        '%d/%m/%Y, %H:%M',
        '%d/%m/%y, %H:%M',
        '%m/%d/%Y, %H:%M',
        '%m/%d/%y, %H:%M',
        '%d/%m/%Y, %H:%M:%S',
        '%d/%m/%y, %H:%M:%S',
        '%d/%m/%Y, %I:%M %p',
        '%d/%m/%y, %I:%M %p',
        '%m/%d/%Y, %I:%M %p',
        '%m/%d/%y, %I:%M %p',
        '%d/%m/%Y, %I:%M:%S %p',
        '%m/%d/%y, %I:%M:%S %p',
    ]
    
    # Clean the timestamp string
    timestamp_str = timestamp_str.strip().replace('\u202f', ' ').replace('\u00a0', ' ')
    
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    return None


def is_system_message(sender: str, message: str) -> bool:
    """Check if a message is a system message (not from a real user)."""
    
    system_indicators = [
        'Messages and calls are end-to-end encrypted',
        'created group',
        'added you',
        'removed you',
        'left the group',
        'changed the subject',
        'changed this group',
        'changed the group',
        'deleted this message',
        'This message was deleted',
        '<Media omitted>',
        'missed voice call',
        'missed video call',
    ]
    
    full_text = f"{sender}: {message}"
    
    for indicator in system_indicators:
        if indicator.lower() in full_text.lower():
            return True
    
    return False


def get_participants(df: pd.DataFrame) -> List[str]:
    """Get list of unique participants from the chat."""
    if df.empty or 'sender' not in df.columns:
        return []
    return sorted(df['sender'].unique().tolist())


def get_chat_stats(df: pd.DataFrame) -> dict:
    """Get basic statistics about the chat."""
    if df.empty:
        return {
            'total_messages': 0,
            'participants': 0,
            'date_range': 'N/A'
        }
    
    stats = {
        'total_messages': len(df),
        'participants': df['sender'].nunique(),
        'date_range': 'N/A'
    }
    
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        valid_dates = df['timestamp'].dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            if min_date and max_date:
                stats['date_range'] = f"{min_date.strftime('%b %d, %Y')} - {max_date.strftime('%b %d, %Y')}"
    
    return stats

