"""LLM utility functions"""
import os
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.7):
    """
    Factory function to create LLM instance.
    
    Args:
        temperature: Model temperature (0.0-1.0)
        
    Returns:
        ChatOpenAI instance configured with GPT-4o
    """
    return ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature
    )