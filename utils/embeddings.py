"""
RAG Module - Embeddings and Gemini Integration
For semantic search and AI-powered chat analysis
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import google.generativeai as genai


def is_rate_limit_error(error_str: str) -> bool:
    """Check if error is related to rate limits."""
    error_lower = error_str.lower()
    return any(x in error_lower for x in ["quota", "limit", "exhausted", "rate", "429", "resource"])


class ChatRAG:
    """RAG system for WhatsApp chat analysis using Gemini with smart fallback."""
    
    # Model priority: try 2.5 first, fallback to 2.0
    PRIMARY_MODEL = 'gemini-2.5-flash'
    FALLBACK_MODEL = 'gemini-2.0-flash'
    
    # Cooldown: how long to wait before trying primary model again after failure
    COOLDOWN_MINUTES = 5
    
    # Class-level tracking (shared across instances)
    _primary_failed_at = None
    
    def __init__(self, api_key: str):
        """Initialize with Gemini API key."""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.primary_model = genai.GenerativeModel(self.PRIMARY_MODEL)
        self.fallback_model = genai.GenerativeModel(self.FALLBACK_MODEL)
        self.df = None
        self.embeddings = None
        self.embedding_model = None
    
    def _should_try_primary(self) -> bool:
        """Check if we should try the primary model or skip to fallback."""
        if ChatRAG._primary_failed_at is None:
            return True
        
        # Check if cooldown period has passed
        time_since_failure = datetime.now() - ChatRAG._primary_failed_at
        if time_since_failure > timedelta(minutes=self.COOLDOWN_MINUTES):
            # Cooldown passed, reset and try primary again
            ChatRAG._primary_failed_at = None
            return True
        
        return False
    
    def _mark_primary_failed(self):
        """Mark that primary model failed (rate limited)."""
        ChatRAG._primary_failed_at = datetime.now()
    
    def load_chat(self, df: pd.DataFrame):
        """Load chat DataFrame and prepare for querying."""
        self.df = df.copy()
        
        # Create combined text for each message (for context)
        self.df['combined'] = self.df.apply(
            lambda row: f"[{row['timestamp']}] {row['sender']}: {row['message']}" 
            if pd.notna(row['timestamp']) 
            else f"{row['sender']}: {row['message']}", 
            axis=1
        )
    
    def _generate_with_fallback(self, prompt: str) -> Tuple[str, bool]:
        """
        Try primary model first (if not in cooldown), fallback to secondary if rate limited.
        Returns: (response_text, used_fallback)
        """
        
        # Check if we should try primary or go straight to fallback
        if self._should_try_primary():
            try:
                response = self.primary_model.generate_content(prompt)
                return response.text, False
            except Exception as e:
                if is_rate_limit_error(str(e)):
                    # Mark primary as failed, start cooldown
                    self._mark_primary_failed()
                    # Continue to fallback below
                else:
                    return "😅 Couldn't get a response right now. Please try again in a moment!", False
        
        # Use fallback model (either primary in cooldown or primary just failed)
        try:
            response = self.fallback_model.generate_content(prompt)
            # Add funny note if we're using fallback due to rate limit
            if ChatRAG._primary_failed_at is not None:
                funny_msg = "💀 *whispers* the dev is lowkey broke so I'm using the budget AI... but here's your answer:\n\n"
                return funny_msg + response.text, True
            return response.text, True
        except Exception as e2:
            if is_rate_limit_error(str(e2)):
                return "💀 Bruh... the dev who made this is BROKE broke! Like, instant-noodles-for-dinner broke. 🍜 Wanna Venmo some API credits? No? Okay, just wait a minute and try again. 😭💸", True
            return "😅 Couldn't get a response right now. Please try again in a moment!", True
    
    def _simple_search(self, query: str, top_k: int = 20) -> List[str]:
        """Simple keyword-based search as fallback."""
        if self.df is None or self.df.empty:
            return []
        
        query_words = query.lower().split()
        
        # Score each message based on word matches
        scores = []
        for idx, row in self.df.iterrows():
            message = str(row['message']).lower()
            score = sum(1 for word in query_words if word in message)
            scores.append((idx, score))
        
        # Sort by score and get top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in scores[:top_k] if score > 0]
        
        # If no matches, return recent messages
        if not top_indices:
            top_indices = list(self.df.index[-top_k:])
        
        return self.df.loc[top_indices, 'combined'].tolist()
    
    def query(self, question: str, top_k: int = 20) -> str:
        """
        Query the chat with a natural language question.
        
        Args:
            question: The question to ask about the chat
            top_k: Number of relevant messages to retrieve
        
        Returns:
            AI-generated response based on chat context
        """
        if self.df is None or self.df.empty:
            return "No chat data loaded. Please upload a WhatsApp chat export first."
        
        # Get relevant context using simple search
        relevant_messages = self._simple_search(question, top_k)
        
        if not relevant_messages:
            return "I couldn't find relevant messages for your question."
        
        # Build context
        context = "\n".join(relevant_messages)
        
        # Create prompt
        prompt = f"""You are a helpful assistant analyzing a WhatsApp group chat. 
Based on the following chat messages, answer the user's question in a fun, engaging way.
Be specific and reference actual messages/people when possible.

CHAT CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the chat messages provided
- Be specific - mention names, dates, and quote messages when relevant
- If the answer isn't clear from the context, say so
- Keep the tone fun and casual, like you're gossiping with a friend 🕵️
- Use emojis sparingly for fun

YOUR RESPONSE:"""
        
        # Generate response with fallback
        response_text, _ = self._generate_with_fallback(prompt)
        return response_text
    
    def get_summary(self) -> str:
        """Get a fun summary of the chat."""
        if self.df is None or self.df.empty:
            return "No chat data to summarize."
        
        # Get sample of messages across the chat
        sample_size = min(50, len(self.df))
        sample = self.df.sample(n=sample_size) if len(self.df) > sample_size else self.df
        
        context = "\n".join(sample['combined'].tolist())
        
        prompt = f"""Analyze this WhatsApp group chat and provide a fun, gossipy summary.

CHAT SAMPLE:
{context}

Provide:
1. 🎭 Group vibe (what kind of group is this?)
2. 👑 Who seems to be the most active/dominant?
3. 🔥 Any hot topics or recurring themes?
4. 😂 Any funny moments or inside jokes you noticed?

Keep it fun and engaging! Use emojis."""
        
        # Generate response with fallback
        response_text, _ = self._generate_with_fallback(prompt)
        return response_text
