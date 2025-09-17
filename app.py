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
from src.modules.prompt_lab import PromptLab
from src.modules.ai_agent import AIAgent
from src.modules.mcp_server import render_mcp_interface
from src.modules.fine_tuning import FineTuningDemo


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
    st.markdown("*Professional AI chat interface with PDF context, detailed analytics, and prompt engineering lab*")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Chat Assistant", "🧪 Prompt Lab", "🤖 AI Agent", "🔌 MCP Server", "🔌 MCP Client", "🎯 Fine-tuning"])
    
    with tab1:
        # Render sidebar and get current settings
        selected_model, parameters = UIComponents.render_sidebar()
        
        # Render main chat interface
        UIComponents.render_chat_interface()
        
        # Handle chat input
        handle_chat_input(selected_model, parameters)
    
    with tab2:
        # Initialize prompt lab
        if "prompt_lab" not in st.session_state:
            st.session_state.prompt_lab = PromptLab()
        
        # Render prompt lab interface
        st.session_state.prompt_lab.render_prompt_lab_interface()
    
    with tab3:
        # Initialize AI agent
        if "ai_agent" not in st.session_state:
            st.session_state.ai_agent = AIAgent()
        
        # Render AI agent interface
        st.session_state.ai_agent.render_agent_interface()
    
    with tab4:
        # Render MCP server interface
        render_mcp_interface()
    
    with tab5:
        # Render MCP client interface
        st.info("🔌 **MCP Client Tab** - This tab demonstrates how external applications can connect to our MCP server.")
        st.markdown("""
        **To test the MCP client:**
        1. Run the headless MCP server: `python mcp_server.py`
        2. Run the MCP client UI: `streamlit run mcp_client_ui.py`
        3. Or use the command-line client: `python mcp_client.py`
        
        **What this demonstrates:**
        - **Protocol Compliance**: Proper MCP protocol implementation
        - **Tool Discovery**: AI can discover available capabilities
        - **Structured Communication**: Standardized request/response formats
        - **Resource Caching**: Results are cached and retrievable
        - **Prompt Templates**: AI can get prompt templates for different tasks
        """)
        
        st.markdown("---")
        st.subheader("📋 MCP Server Commands")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Start MCP Server:**")
            st.code("python mcp_server.py", language="bash")
            
            st.markdown("**Test with Client:**")
            st.code("python mcp_client.py", language="bash")
        
        with col2:
            st.markdown("**Start Client UI:**")
            st.code("streamlit run mcp_client_ui.py", language="bash")
            
            st.markdown("**Server Features:**")
            st.markdown("""
            - ✅ JSON-RPC 2.0 protocol
            - ✅ Tool discovery and calling
            - ✅ Resource caching
            - ✅ Prompt templates
            - ✅ Error handling
            - ✅ Headless operation
            """)
        
        st.markdown("---")
        st.subheader("🔧 MCP Protocol Details")
        
        with st.expander("📋 Available MCP Methods", expanded=True):
            st.markdown("""
            **Core Methods:**
            - `initialize` - Initialize MCP connection
            - `tools/list` - List available tools
            - `tools/call` - Execute a tool
            - `resources/list` - List cached resources
            - `resources/read` - Read cached resource
            - `prompts/list` - List available prompts
            - `prompts/get` - Get prompt template
            """)
        
        with st.expander("🛠️ Available Tools", expanded=False):
            st.markdown("""
            **get_weather:**
            - Parameters: location (string), weather_type (enum)
            - Returns: Temperature, condition, AI summary, recommendations
            
            **get_news:**
            - Parameters: category (enum), country (enum), article_count (int)
            - Returns: Articles, AI summary, key themes
            """)
        
        with st.expander("📝 Available Prompts", expanded=False):
            st.markdown("""
            **weather_summary:**
            - Arguments: city, window
            - Generates AI weather summary prompts
            
            **news_brief:**
            - Arguments: topic, region
            - Generates AI news briefing prompts
            """)
        
        st.markdown("---")
        st.subheader("🎯 Business Value")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **For AI Developers:**
            - Standardized tool integration
            - No manual API integration
            - Automatic capability discovery
            - Consistent error handling
            """)
        
        with col2:
            st.markdown("""
            **For Enterprises:**
            - Tool marketplace compatibility
            - Cross-platform AI integration
            - Rapid capability deployment
            - Future-proof architecture
            """)
    
    with tab6:
        # Initialize fine-tuning demo
        if "fine_tuning_demo" not in st.session_state:
            st.session_state.fine_tuning_demo = FineTuningDemo()
        
        # Render fine-tuning interface
        st.session_state.fine_tuning_demo.render_interface()


def handle_chat_input(selected_model: str, parameters: dict):
    """Handle user chat input and generate AI response"""
    
    # Store selected model in session state for prompt lab access (avoid key conflict)
    if "current_model" not in st.session_state or st.session_state.current_model != selected_model:
        st.session_state.current_model = selected_model
    
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
                pdf_context, chunks_info = SessionManager.get_pdf_context(prompt)
                
                chatbot = st.session_state.chatbot
                evaluator = st.session_state.response_evaluator
                
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
                
                # Evaluate the response
                evaluation = evaluator.evaluate_response(
                    query=prompt,
                    response=response,
                    context=pdf_context,
                    pdf_chunks_info=chunks_info
                )
                
                # Add evaluation and chunks info to metadata
                metadata["evaluation"] = evaluation
                metadata["chunks_info"] = chunks_info
                
                # Display response
                st.markdown(response)
                
                # Display metadata with evaluation
                UIComponents.display_response_metadata(metadata, evaluation, chunks_info)
                
                # Update session stats
                SessionManager.update_conversation_stats(metadata)
                
                # Store metadata
                SessionManager.add_response_metadata(metadata)
        
        # Add assistant response to chat
        SessionManager.add_message("assistant", response)


if __name__ == "__main__":
    main()
