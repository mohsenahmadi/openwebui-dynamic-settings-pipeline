# Pipeline Name: DynamicCategoryLLMAdjuster
# Description: Dynamically adjusts LLM settings based on request category

from typing import Dict, List
import re

# Define category settings
SETTINGS = {
    "Creative Writing": {
        "temperature": 1.0,
        "top_k": 75,
        "top_p": 0.9,
        "max_tokens": 750,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.6,
        "stop_sequences": []
    },
    "Technical Writing": {
        "temperature": 0.2,
        "top_k": 15,
        "top_p": 0.5,
        "max_tokens": 300,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
        "stop_sequences": ["\n\n", "```"]
    },
    "Business Documents": {
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.7,
        "max_tokens": 200,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.5,
        "stop_sequences": ["Best regards", "Sincerely"]
    },
    "Conversational Responses": {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.8,
        "max_tokens": 150,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.4,
        "stop_sequences": ["\n"]
    },
    "Summarization": {
        "temperature": 0.4,
        "top_k": 25,
        "top_p": 0.6,
        "max_tokens": 100,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.2,
        "stop_sequences": ["\n", "."]
    }
}

# Category detection function
def detect_category(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in ["write a story", "novel", "poem", "creative"]):
        return "Creative Writing"
    elif any(keyword in prompt_lower for keyword in ["code", "program", "technical", "document"]):
        return "Technical Writing"
    elif any(keyword in prompt_lower for keyword in ["email", "report", "proposal", "business"]):
        return "Business Documents"
    elif any(keyword in prompt_lower for keyword in ["summarize", "summary", "abstract"]):
        return "Summarization"
    else:
        return "Conversational Responses"  # Default for general queries

# Main pipeline function
def pipeline(prompt: str, llm) -> str:
    # Detect category
    category = detect_category(prompt)
    settings = SETTINGS[category]

    # Configure LLM with dynamic settings
    llm.temperature = settings["temperature"]
    llm.top_k = settings["top_k"]
    llm.top_p = settings["top_p"]
    llm.max_tokens = settings["max_tokens"]
    llm.frequency_penalty = settings["frequency_penalty"]
    llm.presence_penalty = settings["presence_penalty"]
    llm.stop_sequences = settings["stop_sequences"]

    # Generate response
    response = llm.generate(prompt)

    # Append category info to response
    info = (
        f"\n\n[Note: Your request was categorized as '{category}'. "
        f"This response was generated with the following settings: "
        f"temperature={settings['temperature']}, top_k={settings['top_k']}, "
        f"top_p={settings['top_p']}, max_tokens={settings['max_tokens']}, "
        f"frequency_penalty={settings['frequency_penalty']}, "
        f"presence_penalty={settings['presence_penalty']}, "
        f"stop_sequences={settings['stop_sequences']}]"
    )
    return response + info

# Pipeline metadata
PIPELINE_METADATA = {
    "name": "DynamicCategoryLLMAdjuster",
    "description": "Adjusts LLM settings dynamically based on request category",
    "input_type": str,
    "output_type": str
}
