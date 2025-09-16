"""
Prompt Lab Module
Advanced prompt engineering playground with templates and testing functionality
"""

import streamlit as st
from typing import Dict, List, Any
from datetime import datetime


class PromptTemplate:
    """Class to represent a prompt template"""
    
    def __init__(self, name: str, description: str, system_prompt: str, user_prompt: str, use_case: str):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.use_case = use_case


class PromptLab:
    """Advanced prompt engineering laboratory"""
    
    def __init__(self):
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        """Load default prompt templates"""
        templates = {}
        
        # Summarization Template
        templates["summarization"] = PromptTemplate(
            name="📝 Document Summarization",
            description="Generate concise summaries of long documents while preserving key information",
            system_prompt="""You are an expert document analyst specializing in creating comprehensive yet concise summaries. Your task is to:

1. Extract the main themes and key points from the document
2. Preserve important details, facts, and conclusions
3. Organize information in a logical, easy-to-follow structure
4. Maintain the original tone and intent of the document
5. Highlight any critical insights or recommendations

Format your summary with clear sections and bullet points where appropriate.""",
            user_prompt="Please provide a comprehensive summary of the following document, focusing on the main themes, key findings, and important details:\n\n{context}",
            use_case="Document Analysis"
        )
        
        # Fact Checking Template
        templates["fact_check"] = PromptTemplate(
            name="🔍 Fact Verification",
            description="Verify claims and statements against provided document context",
            system_prompt="""You are a meticulous fact-checker with expertise in document analysis. Your responsibilities include:

1. Carefully examine each claim against the provided source material
2. Identify statements that are supported, contradicted, or not addressed by the context
3. Provide specific evidence from the text to support your assessments
4. Note any potential ambiguities or areas requiring clarification
5. Maintain objectivity and avoid speculation beyond the provided information

Format your response with clear verdicts: SUPPORTED, CONTRADICTED, or NOT FOUND, followed by relevant evidence.""",
            user_prompt="Please fact-check the following statements against the provided document. For each claim, indicate whether it is supported, contradicted, or not addressed by the source material:\n\nClaims to verify:\n{user_claims}\n\nSource Document:\n{context}",
            use_case="Information Verification"
        )
        
        # Matching Score Template
        templates["matching"] = PromptTemplate(
            name="🎯 Content Matching Analysis",
            description="Calculate similarity and matching scores between two pieces of content",
            system_prompt="""You are an expert content analyst specializing in similarity assessment and matching analysis. Your expertise includes:

1. Analyzing content overlap and semantic similarity
2. Identifying matching concepts, skills, requirements, and themes
3. Calculating objective matching scores based on multiple criteria
4. Providing detailed explanations for scoring decisions
5. Highlighting both strengths and gaps in alignment

Provide a matching score from 0-100 and break down your analysis into specific categories.""",
            user_prompt="Please analyze the similarity and matching score between the reference content and the comparison content. Provide a detailed breakdown of how well they align:\n\nReference Content (e.g., Job Description):\n{context}\n\nComparison Content (e.g., Resume):\n{comparison_content}\n\nProvide a matching score (0-100) and detailed analysis of alignment in key areas.",
            use_case="Content Comparison"
        )
        
        # Question Answering Template
        templates["qa"] = PromptTemplate(
            name="❓ Question Answering",
            description="Answer specific questions based on document context",
            system_prompt="""You are a knowledgeable assistant specializing in extracting specific information from documents. Your approach includes:

1. Carefully reading and understanding the provided context
2. Providing accurate, direct answers to specific questions
3. Citing relevant sections of the document when possible
4. Acknowledging when information is not available in the context
5. Offering additional relevant insights when appropriate

Be precise, helpful, and always ground your answers in the provided material.""",
            user_prompt="Based on the following document, please answer these specific questions. If an answer cannot be found in the context, please indicate this clearly:\n\nQuestions:\n{questions}\n\nDocument Context:\n{context}",
            use_case="Information Extraction"
        )
        
        # Analysis Template
        templates["analysis"] = PromptTemplate(
            name="📊 Deep Content Analysis",
            description="Perform comprehensive analysis including themes, sentiment, and insights",
            system_prompt="""You are a senior content analyst with expertise in comprehensive document analysis. Your analysis should include:

1. **Thematic Analysis**: Identify main themes, topics, and recurring patterns
2. **Sentiment Analysis**: Assess the overall tone and emotional context
3. **Structural Analysis**: Examine how information is organized and presented
4. **Key Insights**: Extract meaningful conclusions and implications
5. **Recommendations**: Suggest actionable next steps or considerations

Provide a thorough, multi-dimensional analysis that reveals both obvious and subtle aspects of the content.""",
            user_prompt="Please conduct a comprehensive analysis of the following document. Include thematic analysis, sentiment assessment, structural evaluation, and key insights:\n\n{context}",
            use_case="Content Analysis"
        )
        
        return templates
    
    def get_template_names(self) -> List[str]:
        """Get list of available template names"""
        return list(self.templates.keys())
    
    def get_template(self, template_id: str) -> PromptTemplate:
        """Get a specific template by ID"""
        return self.templates.get(template_id)
    
    def render_prompt_lab_interface(self):
        """Render the complete prompt lab interface"""
        st.header("🧪 Prompt Engineering Lab")
        st.markdown("*Experiment with different prompting techniques and test AI responses*")
        
        # Check if PDF is loaded
        if "pdf_data" not in st.session_state:
            st.warning("⚠️ Please upload and process a PDF in the Chat tab first to use the Prompt Lab.")
            st.info("The Prompt Lab uses your uploaded PDF as context for prompt experiments.")
            return
        
        pdf_data = st.session_state.pdf_data
        st.success(f"📄 **Active PDF**: {pdf_data['filename']} ({pdf_data['num_chunks']} chunks)")
        
        st.markdown("---")
        
        # Template Selection Section
        st.subheader("📋 Prompt Templates")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            template_options = ["custom"] + self.get_template_names()
            template_labels = {
                "custom": "🎨 Custom Prompt (Create Your Own)",
                "summarization": "📝 Document Summarization", 
                "fact_check": "🔍 Fact Verification",
                "matching": "🎯 Content Matching Analysis",
                "qa": "❓ Question Answering",
                "analysis": "📊 Deep Content Analysis"
            }
            
            selected_template = st.selectbox(
                "Choose a prompt template or create custom:",
                template_options,
                format_func=lambda x: template_labels.get(x, x),
                key="prompt_template_selector"
            )
        
        with col2:
            if selected_template != "custom":
                template = self.get_template(selected_template)
                st.info(f"**Use Case**: {template.use_case}\n\n{template.description}")
        
        st.markdown("---")
        
        # Prompt Configuration Section
        if selected_template == "custom":
            self._render_custom_prompt_section()
        else:
            self._render_template_prompt_section(selected_template)
        
        st.markdown("---")
        
        # Results Section
        if "prompt_lab_results" in st.session_state and st.session_state.prompt_lab_results:
            self._render_results_section()
    
    def _render_custom_prompt_section(self):
        """Render custom prompt creation section"""
        st.subheader("🎨 Custom Prompt Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**System Prompt** *(Instructions for the AI)*")
            system_prompt = st.text_area(
                "System Prompt",
                placeholder="You are an expert assistant. Your task is to...",
                height=200,
                key="custom_system_prompt",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**User Prompt** *(Your specific request)*")
            user_prompt = st.text_area(
                "User Prompt", 
                placeholder="Please analyze the following document and...\n\nUse {context} to insert PDF content",
                height=200,
                key="custom_user_prompt",
                label_visibility="collapsed"
            )
        
        # Submit button for custom prompt
        if st.button("🚀 Test Custom Prompt", key="submit_custom_prompt"):
            if system_prompt.strip() and user_prompt.strip():
                self._execute_prompt_test(system_prompt, user_prompt, "Custom Prompt")
            else:
                st.error("Please provide both system and user prompts")
    
    def _render_template_prompt_section(self, template_id: str):
        """Render template-based prompt section"""
        template = self.get_template(template_id)
        
        st.subheader(f"{template.name} Configuration")
        
        # Show template description
        st.info(f"**Description**: {template.description}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**System Prompt** *(Editable)*")
            system_prompt = st.text_area(
                "System Prompt",
                value=template.system_prompt,
                height=200,
                key=f"template_system_prompt_{template_id}"
            )
        
        with col2:
            st.markdown("**User Prompt** *(Editable)*")
            user_prompt = st.text_area(
                "User Prompt",
                value=template.user_prompt,
                height=200,
                key=f"template_user_prompt_{template_id}"
            )
        
        # Special inputs for specific templates
        additional_inputs = self._render_template_specific_inputs(template_id)
        
        # Submit button for template
        if st.button(f"🚀 Test {template.name}", key=f"submit_template_{template_id}"):
            if system_prompt.strip() and user_prompt.strip():
                # Replace placeholders in user prompt
                final_user_prompt = self._process_prompt_placeholders(user_prompt, additional_inputs)
                self._execute_prompt_test(system_prompt, final_user_prompt, template.name)
            else:
                st.error("Please provide both system and user prompts")
    
    def _render_template_specific_inputs(self, template_id: str) -> Dict[str, str]:
        """Render additional inputs specific to certain templates"""
        additional_inputs = {}
        
        if template_id == "fact_check":
            st.markdown("**Additional Input**")
            additional_inputs["user_claims"] = st.text_area(
                "Claims to verify:",
                placeholder="Enter the statements you want to fact-check against the PDF...",
                height=100,
                key=f"fact_check_claims_{template_id}"
            )
        
        elif template_id == "matching":
            st.markdown("**Additional Input**")
            additional_inputs["comparison_content"] = st.text_area(
                "Comparison Content (e.g., Resume, Proposal):",
                placeholder="Enter the content you want to compare against the PDF...",
                height=150,
                key=f"matching_content_{template_id}"
            )
        
        elif template_id == "qa":
            st.markdown("**Additional Input**")
            additional_inputs["questions"] = st.text_area(
                "Questions to answer:",
                placeholder="Enter your specific questions about the document...",
                height=100,
                key=f"qa_questions_{template_id}"
            )
        
        return additional_inputs
    
    def _process_prompt_placeholders(self, user_prompt: str, additional_inputs: Dict[str, str]) -> str:
        """Replace placeholders in user prompt with actual content"""
        # Get PDF context
        pdf_data = st.session_state.pdf_data
        context = pdf_data["full_text"]
        
        # Replace placeholders
        processed_prompt = user_prompt.replace("{context}", context)
        
        for key, value in additional_inputs.items():
            processed_prompt = processed_prompt.replace(f"{{{key}}}", value)
        
        return processed_prompt
    
    def _execute_prompt_test(self, system_prompt: str, user_prompt: str, template_name: str):
        """Execute the prompt test and store results"""
        if "chatbot" not in st.session_state:
            st.error("Chatbot not initialized. Please check your API key.")
            return
        
        with st.spinner(f"🧪 Testing {template_name}..."):
            try:
                # Prepare messages for the API call
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Get response from chatbot
                chatbot = st.session_state.chatbot
                response, metadata = chatbot.get_response(
                    messages=messages,
                    model=st.session_state.get("selected_model", "gpt-4o"),
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Evaluate the response
                evaluator = st.session_state.response_evaluator
                evaluation = evaluator.evaluate_response(
                    query=user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
                    response=response,
                    context="",
                    pdf_chunks_info=[]
                )
                
                # Store results
                result = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "template_name": template_name,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": response,
                    "metadata": metadata,
                    "evaluation": evaluation
                }
                
                st.session_state.prompt_lab_results = result
                st.success(f"✅ {template_name} test completed!")
                
            except Exception as e:
                st.error(f"Error executing prompt test: {str(e)}")
    
    def _render_results_section(self):
        """Render the results of prompt testing"""
        st.subheader("📊 Test Results")
        
        results = st.session_state.prompt_lab_results
        
        # Results header
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Template", results["template_name"])
        with col2:
            st.metric("Test Time", results["timestamp"])
        with col3:
            overall_score = results["evaluation"]["metrics"].get("overall_score", 0)
            st.metric("Quality Score", f"{overall_score}/10")
        
        # Response content
        st.subheader("🤖 AI Response")
        st.markdown(results["response"])
        
        # Detailed analysis in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Evaluation", "🔧 Prompts Used", "📈 Metrics", "💾 Export"])
        
        with tab1:
            self._render_evaluation_tab(results["evaluation"])
        
        with tab2:
            self._render_prompts_tab(results)
        
        with tab3:
            self._render_metrics_tab(results["metadata"])
        
        with tab4:
            self._render_export_tab(results)
    
    def _render_evaluation_tab(self, evaluation: Dict):
        """Render evaluation details tab"""
        metrics = evaluation.get("metrics", {})
        
        # Overall score
        score = metrics.get("overall_score", 0)
        color = "green" if score >= 8 else "orange" if score >= 6 else "red"
        st.markdown(f"<h3 style='text-align: center; color: {color}'>Overall Quality Score: {score}/10</h3>", 
                   unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Relevance", f"{metrics.get('relevance_score', 0)}/10")
            st.metric("Completeness", f"{metrics.get('completeness_score', 0)}/10")
            st.metric("Coherence", f"{metrics.get('coherence_score', 0)}/10")
        
        with col2:
            st.metric("Readability", f"{metrics.get('readability_score', 0)}/10")
            st.metric("Word Count", metrics.get('word_count', 0))
            st.metric("Sentences", metrics.get('sentence_count', 0))
    
    def _render_prompts_tab(self, results: Dict):
        """Render prompts used tab"""
        st.subheader("System Prompt")
        st.code(results["system_prompt"], language="text")
        
        st.subheader("User Prompt")
        st.code(results["user_prompt"], language="text")
    
    def _render_metrics_tab(self, metadata: Dict):
        """Render technical metrics tab"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Response Time", f"{metadata.get('response_time', 0):.2f}s")
            st.metric("Model Used", metadata.get('model', 'N/A'))
            st.metric("Temperature", metadata.get('temperature', 0))
        
        with col2:
            st.metric("Total Tokens", metadata.get('total_tokens', 0))
            st.metric("Input Tokens", metadata.get('prompt_tokens', 0))
            st.metric("Output Tokens", metadata.get('completion_tokens', 0))
        
        st.metric("Total Cost", f"${metadata.get('total_cost', 0):.6f}")
    
    def _render_export_tab(self, results: Dict):
        """Render export functionality tab"""
        st.subheader("Export Test Results")
        
        # Create export content
        export_content = f"""# Prompt Lab Test Results
        
## Test Information
- **Template**: {results['template_name']}
- **Timestamp**: {results['timestamp']}
- **Quality Score**: {results['evaluation']['metrics'].get('overall_score', 0)}/10

## System Prompt
{results['system_prompt']}

## User Prompt
{results['user_prompt']}

## AI Response
{results['response']}

## Evaluation Metrics
- **Relevance**: {results['evaluation']['metrics'].get('relevance_score', 0)}/10
- **Completeness**: {results['evaluation']['metrics'].get('completeness_score', 0)}/10
- **Coherence**: {results['evaluation']['metrics'].get('coherence_score', 0)}/10
- **Readability**: {results['evaluation']['metrics'].get('readability_score', 0)}/10

## Technical Details
- **Model**: {results['metadata'].get('model', 'N/A')}
- **Tokens**: {results['metadata'].get('total_tokens', 0)}
- **Cost**: ${results['metadata'].get('total_cost', 0):.6f}
- **Response Time**: {results['metadata'].get('response_time', 0):.2f}s
"""
        
        st.download_button(
            "📄 Download Test Results",
            data=export_content,
            file_name=f"prompt_lab_test_{results['timestamp'].replace(':', '')}.md",
            mime="text/markdown",
            key="export_prompt_results"
        )
        
        if st.button("🗑️ Clear Results", key="clear_prompt_results"):
            del st.session_state.prompt_lab_results
            st.rerun()
