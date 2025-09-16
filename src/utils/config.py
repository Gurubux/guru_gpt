"""
Configuration and Constants
Contains all application constants, model configurations, and settings
"""

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

# Default parameter values
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Parameter ranges and constraints
PARAM_RANGES = {
    "temperature": {"min": 0.0, "max": 2.0, "step": 0.1},
    "max_tokens": {"min": 100, "max": 4000, "step": 100},
    "top_p": {"min": 0.0, "max": 1.0, "step": 0.05},
    "frequency_penalty": {"min": -2.0, "max": 2.0, "step": 0.1},
    "presence_penalty": {"min": -2.0, "max": 2.0, "step": 0.1}
}

# Parameter descriptions
PARAM_DESCRIPTIONS = {
    "temperature": "Higher values make output more random, lower values more focused",
    "max_tokens": "Maximum number of tokens in the response",
    "top_p": "Nucleus sampling parameter",
    "frequency_penalty": "Reduce repetition of frequent tokens",
    "presence_penalty": "Encourage discussion of new topics"
}

# UI Configuration
UI_CONFIG = {
    "page_title": "Advanced GPT Chat Assistant",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# File upload configuration
UPLOAD_CONFIG = {
    "max_file_size": 10,  # MB
    "allowed_extensions": [".pdf", ".txt", ".docx"],
    "chunk_size": 1000,  # characters per chunk
    "chunk_overlap": 200  # overlap between chunks
}
