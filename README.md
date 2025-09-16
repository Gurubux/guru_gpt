# Advanced GPT Chat Assistant

A professional, modular chat application with PDF context support built with Streamlit and OpenAI's latest GPT models. Perfect for AI Engineer interview preparation and demonstrating advanced LLM integration skills.

## Features

- 🤖 **Multi-Model Support**: Choose from GPT-4o, GPT-4o Mini, and GPT-4 Turbo
- 📄 **PDF Context Integration**: Upload and chat with PDF documents
- 🧪 **Prompt Engineering Lab**: Advanced prompt testing with professional templates
- 💬 **Interactive Streamlit UI**: Modern, professional chat interface with tabs
- 📝 **Chat History Management**: Persistent conversations with export functionality
- ⚙️ **Advanced Parameters**: Temperature, max tokens, top-p, frequency/presence penalties
- 🔐 **Secure API Key Management**: Environment variable based configuration
- 📊 **Token Analytics**: Real-time token usage and cost tracking
- ⏱️ **Response Metrics**: Response time, model info, and detailed analytics
- 💰 **Cost Calculation**: Accurate cost tracking per model and conversation
- 📈 **Session Statistics**: Cumulative stats for tokens, costs, and message counts
- 📋 **Export Features**: Download chat history and prompt test results
- 🏗️ **Modular Architecture**: Clean, maintainable codebase with separation of concerns

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

1. Copy the environment template:
   ```bash
   cp env_template.txt .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
guru_gpt/
├── app.py                          # Main Streamlit application (clean & modular)
├── src/                            # Source code modules
│   ├── modules/                    # Core functionality modules
│   │   ├── chatbot.py             # GPT chatbot with context support
│   │   ├── pdf_processor.py       # PDF upload and text extraction
│   │   └── ui_components.py       # Streamlit UI components
│   └── utils/                      # Utility modules
│       ├── config.py              # Configuration and constants
│       └── session_manager.py     # Session state management
├── requirements.txt                # Python dependencies
├── env_template.txt               # Environment variables template
├── setup.py                       # Automated setup script
└── README.md                      # This file
```

## Usage

### 💬 Chat Assistant Tab
#### Basic Chat
1. Start the application using `streamlit run app.py`
2. Make sure your OpenAI API key is properly configured
3. Choose your preferred GPT model from the sidebar
4. Adjust parameters (temperature, max tokens, etc.) as needed
5. Type your message in the chat input
6. View detailed response analytics for each interaction

#### PDF Context Chat
1. Upload a PDF file using the "📄 PDF Context" section in the sidebar
2. Click "Process PDF" to extract and chunk the text
3. Your subsequent chat messages will use the PDF content as context
4. The AI will answer questions based on both the conversation and PDF content
5. Remove PDF context anytime using the "Remove PDF Context" button

### 🧪 Prompt Lab Tab
#### Using Pre-built Templates
1. Upload and process a PDF first (required for context)
2. Navigate to the "🧪 Prompt Lab" tab
3. Select from professional prompt templates:
   - **📝 Document Summarization**: Generate comprehensive summaries
   - **🔍 Fact Verification**: Verify claims against document content
   - **🎯 Content Matching**: Calculate similarity scores (resume-job matching)
   - **❓ Question Answering**: Extract specific information
   - **📊 Deep Content Analysis**: Comprehensive thematic analysis
4. Edit the system and user prompts as needed
5. Add template-specific inputs (claims, comparison content, questions)
6. Click "🚀 Test" to execute and analyze results

#### Creating Custom Prompts
1. Select "🎨 Custom Prompt (Create Your Own)"
2. Write your system prompt (AI instructions)
3. Write your user prompt (use `{context}` for PDF content)
4. Test your prompt and analyze the results
5. Export results for documentation and learning

## Features Explained

### Multi-Model Support
- **GPT-4o**: Most advanced model for complex tasks
- **GPT-4o Mini**: Faster and more cost-effective option
- **GPT-4 Turbo**: Previous generation, still very capable
- Real-time cost estimation for each model

### Advanced Parameters
- **Temperature** (0.0-2.0): Controls randomness vs focus
- **Max Tokens** (100-4000): Response length limit
- **Top P** (0.0-1.0): Nucleus sampling for diversity
- **Frequency Penalty** (-2.0 to 2.0): Reduces repetition
- **Presence Penalty** (-2.0 to 2.0): Encourages new topics

### Analytics & Monitoring
- **Token Usage**: Input, output, and total token counts
- **Cost Tracking**: Per-message and session-wide cost calculation
- **Response Time**: Performance monitoring
- **Session Stats**: Cumulative metrics across conversations

### PDF Context Integration
- **File Upload**: Support for PDF documents up to 10MB
- **Text Extraction**: Automatic text extraction from PDF pages
- **Smart Chunking**: Intelligent text chunking with overlap for better context
- **Relevance Matching**: AI selects most relevant PDF chunks for each query
- **Context Management**: Easy addition and removal of PDF context

### Professional UI
- **Model Selection**: Easy switching between GPT models
- **Parameter Controls**: Real-time adjustment with helpful tooltips
- **Response Details**: Expandable metadata for each AI response
- **Export Functionality**: Download conversations as text files
- **Session Management**: Clear history and reset statistics

### Prompt Engineering Lab
- **Professional Templates**: 5+ pre-built prompt templates for common use cases
- **Editable Prompts**: Modify system and user prompts in real-time
- **Template-Specific Inputs**: Custom fields for fact-checking, matching, Q&A
- **Custom Prompt Creation**: Build and test your own prompting strategies
- **Comprehensive Analysis**: Quality scoring and detailed evaluation of results
- **Export Functionality**: Save test results and prompt configurations

### Modular Architecture
- **Clean Separation**: Distinct modules for chatbot, PDF processing, UI, and prompt lab
- **Maintainable Code**: Easy to extend and modify individual components
- **Reusable Components**: Modular design enables code reuse
- **Production Ready**: Professional code structure suitable for deployment

## Interview Preparation Value

This professional application demonstrates key AI Engineer skills:

### Technical Skills
- **LLM Integration**: Multi-model API usage with OpenAI's latest models
- **Prompt Engineering**: Professional template system with custom prompt creation
- **Document Processing**: PDF text extraction and intelligent chunking
- **Context Management**: Advanced prompt engineering with document context
- **UI/UX Development**: Professional Streamlit interface with tab navigation
- **Code Architecture**: Modular, maintainable, and scalable codebase
- **Error Handling**: Production-ready error management and validation
- **Performance Monitoring**: Token usage, cost tracking, and response metrics
- **Evaluation Systems**: Comprehensive AI response quality assessment

### Software Engineering Practices
- **Modular Design**: Clean separation of concerns across multiple modules
- **Configuration Management**: Centralized configuration and constants
- **Session Management**: Stateful application with proper state handling
- **Documentation**: Comprehensive documentation and code comments
- **Version Control**: Professional git workflow and commit practices

### Production Readiness
- **Security**: Environment-based API key management
- **Scalability**: Modular architecture supports easy feature addition
- **Monitoring**: Detailed analytics and performance tracking
- **User Experience**: Intuitive interface with helpful tooltips and feedback
- **Export Capabilities**: Data portability and user convenience features

Perfect for showcasing in AI Engineer interviews and technical demonstrations!
