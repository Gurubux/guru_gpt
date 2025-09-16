import streamlit as st
import openai
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Available GPT models (latest 3)
AVAILABLE_MODELS = {
    "gpt-4o": {
        "name": "GPT-4o",
        "description": "Most advanced model, excellent for complex tasks",
        "input_cost_per_1k": 0.005,
        "output_cost_per_1k": 0.015
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini", 
        "description": "Faster and more cost-effective version",
        "input_cost_per_1k": 0.00015,
        "output_cost_per_1k": 0.0006
    },
    "gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "description": "Previous generation, still very capable",
        "input_cost_per_1k": 0.01,
        "output_cost_per_1k": 0.03
    }
}

class GPTChatbot:
    """Advanced GPT chatbot with configurable parameters"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_response(self, 
                    messages: List[Dict[str, str]], 
                    model: str = "gpt-4o",
                    temperature: float = 0.7,
                    max_tokens: int = 1000,
                    top_p: float = 1.0,
                    frequency_penalty: float = 0.0,
                    presence_penalty: float = 0.0) -> Tuple[str, Dict]:
        """Get response from GPT with detailed metadata"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=False
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Extract response details
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate costs
            model_info = AVAILABLE_MODELS.get(model, AVAILABLE_MODELS["gpt-4o"])
            input_cost = (usage.prompt_tokens / 1000) * model_info["input_cost_per_1k"]
            output_cost = (usage.completion_tokens / 1000) * model_info["output_cost_per_1k"]
            total_cost = input_cost + output_cost
            
            metadata = {
                "model": model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "response_time": response_time,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            
            return content, metadata
            
        except Exception as e:
            error_metadata = {
                "error": True,
                "error_message": str(e),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            return f"Error: {str(e)}", error_metadata

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "assistant", "content": "Hello! I'm your AI assistant. Choose your preferred model and settings, then start chatting!"}
        ]
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = GPTChatbot()
    if "conversation_stats" not in st.session_state:
        st.session_state.conversation_stats = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "message_count": 0
        }
    if "response_metadata" not in st.session_state:
        st.session_state.response_metadata = []

def display_response_metadata(metadata: Dict):
    """Display response metadata in an expandable section"""
    if "error" in metadata:
        st.error(f"⚠️ Error at {metadata['timestamp']}: {metadata['error_message']}")
        return
    
    with st.expander("📊 Response Details", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Response Time", f"{metadata['response_time']:.2f}s")
            st.metric("Model", metadata['model'])
            st.metric("Temperature", metadata['temperature'])
        
        with col2:
            st.metric("Total Tokens", metadata['total_tokens'])
            st.metric("Input Tokens", metadata['prompt_tokens'])
            st.metric("Output Tokens", metadata['completion_tokens'])
        
        with col3:
            st.metric("Total Cost", f"${metadata['total_cost']:.6f}")
            st.metric("Input Cost", f"${metadata['input_cost']:.6f}")
            st.metric("Output Cost", f"${metadata['output_cost']:.6f}")

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Advanced GPT Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🤖 Advanced GPT Chat Assistant")
    st.markdown("*Professional AI chat interface with model selection and detailed analytics*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key status
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ OpenAI API Key loaded")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info("Please add your OpenAI API key to the .env file")
        
        st.markdown("---")
        
        # Model Selection
        st.header("🎯 Model Selection")
        selected_model = st.selectbox(
            "Choose GPT Model:",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: f"{AVAILABLE_MODELS[x]['name']} - {AVAILABLE_MODELS[x]['description']}",
            index=0
        )
        
        # Display model info
        model_info = AVAILABLE_MODELS[selected_model]
        st.info(f"**{model_info['name']}**\n{model_info['description']}\n\n"
                f"💰 Input: ${model_info['input_cost_per_1k']}/1K tokens\n"
                f"💰 Output: ${model_info['output_cost_per_1k']}/1K tokens")
        
        st.markdown("---")
        
        # Parameters
        st.header("⚙️ Parameters")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more focused"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Maximum number of tokens in the response"
        )
        
        with st.expander("🔧 Advanced Parameters"):
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="Nucleus sampling parameter"
            )
            
            frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help="Reduce repetition of frequent tokens"
            )
            
            presence_penalty = st.slider(
                "Presence Penalty", 
                min_value=-2.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help="Encourage discussion of new topics"
            )
        
        st.markdown("---")
        
        # Conversation Stats
        st.header("📈 Session Stats")
        stats = st.session_state.conversation_stats
        st.metric("Messages", stats["message_count"])
        st.metric("Total Tokens", stats["total_tokens"])
        st.metric("Total Cost", f"${stats['total_cost']:.6f}")
        
        st.markdown("---")
        
        # Chat controls
        st.header("🗂️ Chat Controls")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "assistant", "content": "Hello! I'm your AI assistant. Choose your preferred model and settings, then start chatting!"}
            ]
            st.session_state.conversation_stats = {
                "total_tokens": 0,
                "total_cost": 0.0,
                "message_count": 0
            }
            st.session_state.response_metadata = []
            st.rerun()
        
        if st.button("📊 Export Chat History"):
            chat_export = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in st.session_state.messages[1:]  # Skip system message
            ])
            st.download_button(
                "💾 Download Chat",
                data=chat_export,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display chat messages (exclude system message)
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages[1:]):  # Skip system message
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata for assistant responses
                if (message["role"] == "assistant" and 
                    i < len(st.session_state.response_metadata) and
                    st.session_state.response_metadata[i] is not None):
                    display_response_metadata(st.session_state.response_metadata[i])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        if not api_key:
            st.error("Please configure your OpenAI API key in the .env file")
            return
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response, metadata = st.session_state.chatbot.get_response(
                    st.session_state.messages,
                    model=selected_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                st.markdown(response)
                
                # Display metadata
                display_response_metadata(metadata)
                
                # Update session stats
                if "error" not in metadata:
                    st.session_state.conversation_stats["total_tokens"] += metadata["total_tokens"]
                    st.session_state.conversation_stats["total_cost"] += metadata["total_cost"] 
                    st.session_state.conversation_stats["message_count"] += 1
                
                # Store metadata
                st.session_state.response_metadata.append(metadata)
        
        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
