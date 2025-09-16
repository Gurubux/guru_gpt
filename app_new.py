"""
Advanced GPT Chat Assistant - Main Application
Clean, modular main application file with PDF chat functionality
"""

import streamlit as st
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import UI_CONFIG
from src.utils.session_manager import SessionManager
from src.modules.ui_components import UIComponents


def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title=UI_CONFIG["page_title"],
        page_icon=UI_CONFIG["page_icon"],
        layout=UI_CONFIG["layout"],
        initial_sidebar_state=UI_CONFIG["initial_sidebar_state"]
    )
    
    # Initialize session state
    SessionManager.initialize_session_state()
    
    # Header
    st.title("🤖 Advanced GPT Chat Assistant")
    st.markdown("*Professional AI chat interface with PDF context and detailed analytics*")
    st.markdown("---")
    
    # Render sidebar and get current settings
    selected_model, parameters = UIComponents.render_sidebar()
    
    # Render main chat interface
    UIComponents.render_chat_interface()
    
    # Handle chat input
    handle_chat_input(selected_model, parameters)


def handle_chat_input(selected_model: str, parameters: dict):
    """Handle user chat input and generate AI response"""
    
    if prompt := st.chat_input("Type your message here..."):
        if not st.session_state.get("api_key"):
            st.error("Please configure your OpenAI API key in the .env file")
            return
        
        if not st.session_state.get("chatbot"):
            st.error("Chatbot not initialized. Please check your API key.")
            return
        
        # Add user message to chat
        SessionManager.add_message("user", prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Get PDF context if available
                pdf_context = SessionManager.get_pdf_context(prompt)
                
                chatbot = st.session_state.chatbot
                
                if pdf_context:
                    # Use PDF context for response
                    response, metadata = chatbot.get_response_with_context(
                        st.session_state.messages,
                        context_text=pdf_context,
                        model=selected_model,
                        **parameters
                    )
                else:
                    # Regular response without PDF context
                    response, metadata = chatbot.get_response(
                        st.session_state.messages,
                        model=selected_model,
                        **parameters
                    )
                
                # Display response
                st.markdown(response)
                
                # Display metadata
                UIComponents.display_response_metadata(metadata)
                
                # Update session stats
                SessionManager.update_conversation_stats(metadata)
                
                # Store metadata
                SessionManager.add_response_metadata(metadata)
        
        # Add assistant response to chat
        SessionManager.add_message("assistant", response)


if __name__ == "__main__":
    main()
