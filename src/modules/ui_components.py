"""
UI Components Module
Contains all Streamlit UI components and display functions
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List
from src.utils.config import AVAILABLE_MODELS, PARAM_RANGES, PARAM_DESCRIPTIONS, DEFAULT_PARAMS


class UIComponents:
    """Handle all UI components and interactions"""
    
    @staticmethod
    def display_response_metadata(metadata: Dict, evaluation: Dict = None, chunks_info: list = None):
        """Display response metadata and evaluation in expandable sections"""
        if "error" in metadata:
            st.error(f"⚠️ Error at {metadata['timestamp']}: {metadata['error_message']}")
            return
        
        # Response Details Section
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
        
        # Evaluation Section
        if evaluation:
            UIComponents.display_evaluation_details(evaluation)
        
        # PDF Context Section
        if chunks_info:
            UIComponents.display_pdf_context_details(chunks_info)
    
    @staticmethod
    def render_sidebar():
        """Render the complete sidebar with all controls"""
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # API Key status
            api_key = st.session_state.get("api_key")
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
                index=0,
                key="selected_model"
            )
            
            # Display model info
            model_info = AVAILABLE_MODELS[selected_model]
            st.info(f"**{model_info['name']}**\n{model_info['description']}\n\n"
                    f"💰 Input: ${model_info['input_cost_per_1k']}/1K tokens\n"
                    f"💰 Output: ${model_info['output_cost_per_1k']}/1K tokens")
            
            st.markdown("---")
            
            # PDF Upload Section
            UIComponents.render_pdf_upload_section()
            
            st.markdown("---")
            
            # Parameters
            parameters = UIComponents.render_parameters_section()
            
            st.markdown("---")
            
            # Conversation Stats
            UIComponents.render_stats_section()
            
            st.markdown("---")
            
            # Chat controls
            UIComponents.render_chat_controls()
            
            return selected_model, parameters
    
    @staticmethod
    def render_pdf_upload_section():
        """Render PDF upload and processing section"""
        st.header("📄 PDF Context")
        
        uploaded_file = st.file_uploader(
            "Upload PDF for context",
            type=['pdf'],
            help="Upload a PDF file to provide context for your chat"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    processor = st.session_state.get("pdf_processor")
                    if processor:
                        pdf_data = processor.process_pdf(uploaded_file)
                        if pdf_data:
                            st.session_state.pdf_data = pdf_data
                            st.success(f"✅ PDF processed: {pdf_data['num_chunks']} chunks, {pdf_data['total_chars']} characters")
        
        # Show current PDF status
        if "pdf_data" in st.session_state:
            pdf_data = st.session_state.pdf_data
            st.info(f"📄 **Active PDF**: {pdf_data['filename']}\n"
                   f"📊 {pdf_data['num_chunks']} chunks, {pdf_data['total_chars']} chars")
            
            # Show PDF content preview
            if st.button("📋 Preview PDF Content"):
                UIComponents.show_pdf_preview(pdf_data)
            
            if st.button("Remove PDF Context"):
                del st.session_state.pdf_data
                st.rerun()
    
    @staticmethod
    def render_parameters_section():
        """Render parameter controls and return parameter values"""
        st.header("⚙️ Parameters")
        
        temperature = st.slider(
            "Temperature",
            min_value=PARAM_RANGES["temperature"]["min"],
            max_value=PARAM_RANGES["temperature"]["max"],
            value=DEFAULT_PARAMS["temperature"],
            step=PARAM_RANGES["temperature"]["step"],
            help=PARAM_DESCRIPTIONS["temperature"]
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=PARAM_RANGES["max_tokens"]["min"],
            max_value=PARAM_RANGES["max_tokens"]["max"],
            value=DEFAULT_PARAMS["max_tokens"],
            step=PARAM_RANGES["max_tokens"]["step"],
            help=PARAM_DESCRIPTIONS["max_tokens"]
        )
        
        with st.expander("🔧 Advanced Parameters"):
            top_p = st.slider(
                "Top P",
                min_value=PARAM_RANGES["top_p"]["min"],
                max_value=PARAM_RANGES["top_p"]["max"],
                value=DEFAULT_PARAMS["top_p"],
                step=PARAM_RANGES["top_p"]["step"],
                help=PARAM_DESCRIPTIONS["top_p"]
            )
            
            frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=PARAM_RANGES["frequency_penalty"]["min"],
                max_value=PARAM_RANGES["frequency_penalty"]["max"],
                value=DEFAULT_PARAMS["frequency_penalty"],
                step=PARAM_RANGES["frequency_penalty"]["step"],
                help=PARAM_DESCRIPTIONS["frequency_penalty"]
            )
            
            presence_penalty = st.slider(
                "Presence Penalty", 
                min_value=PARAM_RANGES["presence_penalty"]["min"],
                max_value=PARAM_RANGES["presence_penalty"]["max"],
                value=DEFAULT_PARAMS["presence_penalty"],
                step=PARAM_RANGES["presence_penalty"]["step"],
                help=PARAM_DESCRIPTIONS["presence_penalty"]
            )
        
        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }
    
    @staticmethod
    def render_stats_section():
        """Render conversation statistics"""
        st.header("📈 Session Stats")
        stats = st.session_state.get("conversation_stats", {"message_count": 0, "total_tokens": 0, "total_cost": 0.0})
        st.metric("Messages", stats["message_count"])
        st.metric("Total Tokens", stats["total_tokens"])
        st.metric("Total Cost", f"${stats['total_cost']:.6f}")
    
    @staticmethod
    def render_chat_controls():
        """Render chat control buttons"""
        st.header("🗂️ Chat Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                UIComponents.clear_chat_history()
        
        with col2:
            if st.button("📊 Export Chat"):
                UIComponents.export_chat_history()
    
    @staticmethod
    def clear_chat_history():
        """Clear chat history and reset stats"""
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
    
    @staticmethod
    def export_chat_history():
        """Export chat history as downloadable file"""
        messages = st.session_state.get("messages", [])
        if len(messages) > 1:  # More than just system message
            chat_export = "\n".join([
                f"{msg['role'].title()}: {msg['content']}" 
                for msg in messages[1:]  # Skip system message
            ])
            st.download_button(
                "💾 Download Chat",
                data=chat_export,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No chat history to export")
    
    @staticmethod
    def render_chat_interface():
        """Render the main chat interface"""
        st.header("💬 Chat")
        
        # Display chat messages (exclude system message)
        messages = st.session_state.get("messages", [])
        response_metadata = st.session_state.get("response_metadata", [])
        
        for i, message in enumerate(messages[1:]):  # Skip system message
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata for assistant responses
                if (message["role"] == "assistant" and 
                    i < len(response_metadata) and
                    response_metadata[i] is not None):
                    # Get evaluation and chunks info if available
                    evaluation = response_metadata[i].get("evaluation")
                    chunks_info = response_metadata[i].get("chunks_info")
                    UIComponents.display_response_metadata(response_metadata[i], evaluation, chunks_info)
    
    @staticmethod
    def display_evaluation_details(evaluation: Dict):
        """Display response evaluation metrics"""
        with st.expander("🎯 Response Evaluation", expanded=False):
            metrics = evaluation.get("metrics", {})
            
            # Overall score prominently displayed
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                score = metrics.get("overall_score", 0)
                color = "green" if score >= 8 else "orange" if score >= 6 else "red"
                st.markdown(f"<h3 style='text-align: center; color: {color}'>Overall Score: {score}/10</h3>", 
                           unsafe_allow_html=True)
            
            # Detailed metrics
            st.subheader("📋 Detailed Metrics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Relevance", f"{metrics.get('relevance_score', 0)}/10")
                st.metric("Completeness", f"{metrics.get('completeness_score', 0)}/10")
                st.metric("Coherence", f"{metrics.get('coherence_score', 0)}/10")
            
            with col2:
                st.metric("Readability", f"{metrics.get('readability_score', 0)}/10")
                st.metric("Word Count", metrics.get('word_count', 0))
                st.metric("Sentences", metrics.get('sentence_count', 0))
            
            # Context metrics if available
            if "context_utilization" in metrics:
                st.subheader("📄 PDF Context Usage")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Context Utilization", f"{metrics.get('context_utilization', 0)}/10")
                    st.metric("Chunks Used", metrics.get('chunks_used', 0))
                
                with col2:
                    st.metric("Avg Chunk Relevance", f"{metrics.get('avg_chunk_relevance', 0):.2f}")
                    st.metric("Max Chunk Relevance", f"{metrics.get('max_chunk_relevance', 0):.2f}")
            
            # Score explanations
            if st.button("📚 What do these scores mean?"):
                UIComponents.show_score_explanations()
    
    @staticmethod
    def display_pdf_context_details(chunks_info: List[Dict]):
        """Display details about PDF chunks used for context"""
        with st.expander("📄 PDF Context Used", expanded=False):
            st.subheader(f"🔍 {len(chunks_info)} Relevant Chunks Selected")
            
            for i, chunk in enumerate(chunks_info, 1):
                with st.container():
                    st.markdown(f"**Chunk {i} (Index: {chunk['chunk_index']})**")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Relevance Score", f"{chunk['relevance_score']:.2f}")
                    with col2:
                        st.metric("Keyword Matches", chunk['keyword_matches'])
                    with col3:
                        st.metric("Overlap Ratio", f"{chunk['overlap_ratio']:.2%}")
                    
                    # Show matching words
                    if chunk['matching_words']:
                        st.write("🔗 **Matching Keywords:**", ", ".join(chunk['matching_words']))
                    
                    # Show chunk preview
                    st.text_area(f"📝 Chunk Preview", 
                                chunk['chunk_preview'], 
                                height=100, 
                                key=f"chunk_preview_{chunk['chunk_index']}")
                    
                    st.markdown("---")
    
    @staticmethod
    def show_pdf_preview(pdf_data: Dict):
        """Show PDF content preview in a modal-like expander"""
        with st.expander("📄 PDF Content Preview", expanded=True):
            st.subheader(f"📄 {pdf_data['filename']}")
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Characters", pdf_data['total_chars'])
            with col2:
                st.metric("Total Chunks", pdf_data['num_chunks'])
            with col3:
                st.metric("Avg Chunk Size", pdf_data['total_chars'] // pdf_data['num_chunks'])
            
            # Show first 1000 characters
            st.subheader("📖 Content Preview (First 1000 characters)")
            preview_text = pdf_data['full_text'][:1000]
            if len(pdf_data['full_text']) > 1000:
                preview_text += "...\n\n[Content truncated for display]"
            
            st.text_area("", preview_text, height=300, disabled=True)
            
            # Show all chunks
            st.subheader("📑 All Chunks")
            for i, chunk in enumerate(pdf_data['chunks']):
                with st.expander(f"Chunk {i+1} ({len(chunk)} characters)"):
                    st.text_area("", chunk, height=150, key=f"full_chunk_{i}", disabled=True)
    
    @staticmethod
    def show_score_explanations():
        """Show explanations for all evaluation metrics"""
        evaluator = st.session_state.get("response_evaluator")
        if evaluator:
            explanations = evaluator.get_score_explanation()
            
            st.subheader("📚 Evaluation Metrics Explained")
            
            for metric, explanation in explanations.items():
                st.write(f"**{metric.replace('_', ' ').title()}**: {explanation}")
            
            st.info("""
            **Score Interpretation:**
            - **9-10**: Excellent - Professional quality response
            - **7-8**: Good - High quality with minor areas for improvement  
            - **5-6**: Fair - Adequate but with notable limitations
            - **3-4**: Poor - Significant issues requiring attention
            - **0-2**: Very Poor - Major problems in multiple areas
            """)
