# GPT-4o Chat Assistant

A simple chat application built with Streamlit and OpenAI's GPT-4o model. Perfect for AI Engineer interview preparation and demonstrating LLM integration skills.

## Features

- 🤖 **Multi-Model Support**: Choose from GPT-4o, GPT-4o Mini, and GPT-4 Turbo
- 💬 **Interactive Streamlit UI**: Modern, professional chat interface
- 📝 **Chat History Management**: Persistent conversations with export functionality
- ⚙️ **Advanced Parameters**: Temperature, max tokens, top-p, frequency/presence penalties
- 🔐 **Secure API Key Management**: Environment variable based configuration
- 📊 **Token Analytics**: Real-time token usage and cost tracking
- ⏱️ **Response Metrics**: Response time, model info, and detailed analytics
- 💰 **Cost Calculation**: Accurate cost tracking per model and conversation
- 📈 **Session Statistics**: Cumulative stats for tokens, costs, and message counts
- 📋 **Export Features**: Download chat history as text files

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
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── env_template.txt    # Environment variables template
└── README.md          # This file
```

## Usage

1. Start the application using `streamlit run app.py`
2. Make sure your OpenAI API key is properly configured
3. Type your message in the chat input
4. The GPT-4o model will respond to your queries
5. Use the sidebar to clear chat history or view model information

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

### Professional UI
- **Model Selection**: Easy switching between GPT models
- **Parameter Controls**: Real-time adjustment with helpful tooltips
- **Response Details**: Expandable metadata for each AI response
- **Export Functionality**: Download conversations as text files
- **Session Management**: Clear history and reset statistics

## Next Steps for Interview Prep

This basic chatbot demonstrates:
- LLM API integration
- UI development with Streamlit
- Session management
- Error handling
- Environment configuration

Consider extending with:
- Different model options
- Custom system prompts
- File upload capabilities
- Chat export functionality
- User authentication
