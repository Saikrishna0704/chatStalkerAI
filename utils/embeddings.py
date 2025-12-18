"""
RAG Module - Embeddings and Gemini Integration
For semantic search and AI-powered chat analysis
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import google.generativeai as genai


def get_friendly_error(error_str: str) -> str:
    """Return a simple user-friendly message - no technical details."""
    error_lower = error_str.lower()
    
    if "quota" in error_lower or "limit" in error_lower or "exhausted" in error_lower or "rate" in error_lower or "429" in error_lower:
        return "💀 Bruh... the dev who made this is BROKE broke! Like, instant-noodles-for-dinner broke. 🍜 Wanna Venmo some API credits? No? Okay, just wait a minute and try again. 😭💸"
    
    return "😅 Couldn't get a response right now. Please try again in a moment!"


class ChatRAG:
    """RAG system for WhatsApp chat analysis using Gemini."""
    
    def __init__(self, api_key: str):
        """Initialize with Gemini API key."""
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.df = None
        self.embeddings = None
        self.embedding_model = None
    
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
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using Gemini's embedding model."""
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return np.array(embeddings)
        except Exception as e:
            print(f"Embedding error: {e}")
            return np.array([])
    
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
        
        try:
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
            
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return get_friendly_error(str(e))
    
    def get_summary(self) -> str:
        """Get a fun summary of the chat."""
        if self.df is None or self.df.empty:
            return "No chat data to summarize."
        
        try:
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
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return get_friendly_error(str(e))
