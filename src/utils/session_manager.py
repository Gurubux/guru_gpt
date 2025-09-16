"""
Session Management Utilities
Handles Streamlit session state initialization and management
"""

import streamlit as st
import os
from dotenv import load_dotenv
from src.modules.chatbot import GPTChatbot
from src.modules.pdf_processor import PDFProcessor
from src.modules.response_evaluator import ResponseEvaluator


class SessionManager:
    """Manage Streamlit session state and initialization"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables"""
        # Load environment variables
        load_dotenv()
        
        # Initialize API key
        if "api_key" not in st.session_state:
            st.session_state.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize messages
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "assistant", "content": "Hello! I'm your AI assistant. Choose your preferred model and settings, then start chatting!"}
            ]
        
        # Initialize chatbot
        if "chatbot" not in st.session_state and st.session_state.api_key:
            st.session_state.chatbot = GPTChatbot(st.session_state.api_key)
        
        # Initialize PDF processor
        if "pdf_processor" not in st.session_state:
            st.session_state.pdf_processor = PDFProcessor()
        
        # Initialize response evaluator
        if "response_evaluator" not in st.session_state:
            st.session_state.response_evaluator = ResponseEvaluator()
        
        # Initialize conversation stats
        if "conversation_stats" not in st.session_state:
            st.session_state.conversation_stats = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "message_count": 0
            }
        
        # Initialize response metadata
        if "response_metadata" not in st.session_state:
            st.session_state.response_metadata = []
    
    @staticmethod
    def update_conversation_stats(metadata: dict):
        """Update conversation statistics with new response metadata"""
        if "error" not in metadata:
            stats = st.session_state.conversation_stats
            stats["total_tokens"] += metadata["total_tokens"]
            stats["total_cost"] += metadata["total_cost"]
            stats["message_count"] += 1
            st.session_state.conversation_stats = stats
    
    @staticmethod
    def add_message(role: str, content: str):
        """Add a message to the conversation"""
        st.session_state.messages.append({"role": role, "content": content})
    
    @staticmethod
    def add_response_metadata(metadata: dict):
        """Add response metadata to the session"""
        st.session_state.response_metadata.append(metadata)
    
    @staticmethod
    def get_pdf_context(query: str = "") -> tuple[str, list]:
        """Get relevant PDF context and chunk info for the current query"""
        if "pdf_data" not in st.session_state:
            return "", []
        
        pdf_data = st.session_state.pdf_data
        processor = st.session_state.pdf_processor
        
        if query:
            # Get relevant chunks based on query with detailed info
            context_text, chunks_info = processor.get_relevant_chunks(pdf_data["chunks"], query)
            return context_text, chunks_info
        else:
            # Return a summary of the full text (first chunk)
            first_chunk = pdf_data["chunks"][0] if pdf_data["chunks"] else ""
            return first_chunk, []
