"""
GPT Chatbot Module
Handles all GPT model interactions and response processing
"""

import openai
import time
from datetime import datetime
from typing import List, Dict, Tuple
from src.utils.config import AVAILABLE_MODELS


class GPTChatbot:
    """Advanced GPT chatbot with configurable parameters"""
    
    def __init__(self, api_key: str):
        """Initialize the chatbot with OpenAI API key"""
        self.client = openai.OpenAI(api_key=api_key)
    
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
    
    def get_response_with_context(self,
                                 messages: List[Dict[str, str]],
                                 context_text: str = None,
                                 **kwargs) -> Tuple[str, Dict]:
        """Get response with additional context (e.g., from PDF)"""
        if context_text:
            # Add context to the system message or create a new context message
            context_message = {
                "role": "system",
                "content": f"Use the following context to help answer questions: {context_text}"
            }
            # Insert context after system message but before conversation
            enhanced_messages = [messages[0], context_message] + messages[1:]
        else:
            enhanced_messages = messages
        
        return self.get_response(enhanced_messages, **kwargs)
