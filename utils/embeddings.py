"""
RAG Module - Embeddings and Gemini Integration
For semantic search and AI-powered chat analysis
"""

import pandas as pd
import numpy as np
from typing import List
import google.generativeai as genai


def is_rate_limit_error(error_str: str) -> bool:
    """Check if error is related to rate limits."""
    error_lower = error_str.lower()
    return any(x in error_lower for x in ["quota", "limit", "exhausted", "rate", "429", "resource"])


class ChatRAG:
    """RAG system for WhatsApp chat analysis using Gemini."""
    
    def __init__(self, api_key: str):
        """Initialize with Gemini API key."""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.df = None
    
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
    
    def _simple_search(self, query: str, top_k: int = 20) -> List[str]:
        """Simple keyword-based search."""
        if self.df is None or self.df.empty:
            return []
        
        query_words = query.lower().split()
        
        scores = []
        for idx, row in self.df.iterrows():
            message = str(row['message']).lower()
            score = sum(1 for word in query_words if word in message)
            scores.append((idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in scores[:top_k] if score > 0]
        
        if not top_indices:
            top_indices = list(self.df.index[-top_k:])
        
        return self.df.loc[top_indices, 'combined'].tolist()
    
    def query(self, question: str, top_k: int = 20) -> str:
        """Query the chat with a natural language question."""
        if self.df is None or self.df.empty:
            return "No chat data loaded. Please upload a WhatsApp chat export first."
        
        try:
            relevant_messages = self._simple_search(question, top_k)
            
            if not relevant_messages:
                return "I couldn't find relevant messages for your question."
            
            context = "\n".join(relevant_messages)
            
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
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            if is_rate_limit_error(str(e)):
                return "💀 Bruh... the dev who made this is BROKE broke! Like, instant-noodles-for-dinner broke. 🍜 Wanna Venmo some API credits? No? Okay, just wait a minute and try again. 😭💸"
            return "😅 Couldn't get a response right now. Please try again in a moment!"
    
    def get_summary(self) -> str:
        """Get a fun summary of the chat."""
        if self.df is None or self.df.empty:
            return "No chat data to summarize."
        
        try:
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
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            if is_rate_limit_error(str(e)):
                return "💀 Bruh... the dev who made this is BROKE broke! Like, instant-noodles-for-dinner broke. 🍜 Wanna Venmo some API credits? No? Okay, just wait a minute and try again. 😭💸"
            return "😅 Couldn't get a response right now. Please try again in a moment!"
