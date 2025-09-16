"""
PDF Processing Module
Handles PDF file upload, text extraction, and chunking for chat context
"""

import streamlit as st
import PyPDF2
from io import BytesIO
from typing import List, Optional, Dict
from src.utils.config import UPLOAD_CONFIG


class PDFProcessor:
    """Handle PDF file processing and text extraction"""
    
    def __init__(self):
        self.chunk_size = UPLOAD_CONFIG["chunk_size"]
        self.chunk_overlap = UPLOAD_CONFIG["chunk_overlap"]
    
    def extract_text_from_pdf(self, uploaded_file) -> Optional[str]:
        """Extract text from uploaded PDF file"""
        try:
            # Read the PDF file
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks for better context management"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk, find a good break point
            if end < len(text):
                # Look for sentence ending or paragraph break
                break_point = text.rfind('.', start, end)
                if break_point == -1:
                    break_point = text.rfind(' ', start, end)
                if break_point != -1:
                    end = break_point + 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else end
        
        return chunks
    
    def process_pdf(self, uploaded_file) -> Optional[Dict]:
        """Process PDF and return structured data"""
        if uploaded_file is None:
            return None
        
        # Check file size
        if uploaded_file.size > UPLOAD_CONFIG["max_file_size"] * 1024 * 1024:
            st.error(f"File size too large. Maximum allowed: {UPLOAD_CONFIG['max_file_size']}MB")
            return None
        
        # Extract text
        text = self.extract_text_from_pdf(uploaded_file)
        if not text:
            return None
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        return {
            "filename": uploaded_file.name,
            "full_text": text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "total_chars": len(text)
        }
    
    def get_relevant_chunks(self, chunks: List[str], query: str, max_chunks: int = 3) -> tuple[str, List[Dict]]:
        """Get most relevant chunks based on keyword matching with detailed scoring"""
        if not chunks or not query:
            return "", []
        
        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            
            # Calculate different scoring metrics
            keyword_overlap = len(query_words.intersection(chunk_words))
            total_query_words = len(query_words)
            total_chunk_words = len(chunk_words)
            
            # Normalized scores
            overlap_ratio = keyword_overlap / total_query_words if total_query_words > 0 else 0
            chunk_density = keyword_overlap / total_chunk_words if total_chunk_words > 0 else 0
            
            # Combined relevance score
            relevance_score = (keyword_overlap * 2) + (overlap_ratio * 10) + (chunk_density * 5)
            
            if keyword_overlap > 0:  # Only include chunks with at least one matching word
                chunk_info = {
                    "chunk_index": i,
                    "chunk_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "full_chunk": chunk,
                    "keyword_matches": keyword_overlap,
                    "overlap_ratio": overlap_ratio,
                    "chunk_density": chunk_density,
                    "relevance_score": relevance_score,
                    "matching_words": list(query_words.intersection(chunk_words))
                }
                scored_chunks.append((relevance_score, chunk_info))
        
        # Sort by score and take top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks_info = [chunk_info for _, chunk_info in scored_chunks[:max_chunks]]
        relevant_chunks_text = [chunk_info["full_chunk"] for chunk_info in top_chunks_info]
        
        return "\n\n".join(relevant_chunks_text), top_chunks_info
    
    def get_pdf_summary(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of the PDF content"""
        if not text:
            return ""
        
        # Simple extractive summary - get first and last parts
        words = text.split()
        if len(words) <= max_length:
            return text
        
        # Take first 60% and last 40% of the summary length
        first_part_length = int(max_length * 0.6)
        last_part_length = max_length - first_part_length
        
        first_part = " ".join(words[:first_part_length])
        last_part = " ".join(words[-last_part_length:])
        
        return f"{first_part}\n\n[...content truncated...]\n\n{last_part}"
