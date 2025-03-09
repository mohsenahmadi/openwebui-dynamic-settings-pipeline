"""
title: CategoryRoutingPipeline
author: YourName
date: 2025-03-09
version: 1.0
requirements: 
"""

from typing import List, Union, Generator, Iterator, Tuple, Dict
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import re

class Pipeline:
    class Valves(BaseModel):
        """Placeholder for pipeline valve configurations."""
        pass

    def __init__(self):
        self.name = "Category Routing Pipeline"

    def detect_category(self, message: str) -> str:
        """Detects the category of the message using regex-based classification."""
        message = message.lower()
        category_mappings = {
            "Creative Writing": r"story|poem|imagine",
            "Technical Writing": r"code|technical|document",
            "Business Writing": r"report|proposal|business",
            "Educational Content": r"explain|teach|learn",
            "Social Media Posts": r"post|tweet|share",
            "Translation": r"translate|language|convert",
            "Summarization": r"summarize|brief|condense",
            "Question Answering": r"what|how|why"
        }

        for category, pattern in category_mappings.items():
            if re.search(pattern, message):
                return category
        return "General"

    def get_model_and_params(self, category: str) -> Tuple[str, Dict[str, float]]:
        """Maps detected category to an appropriate LLM model with fine-tuned parameters."""
        model_mappings = {
            "Creative Writing": ("anthropic/claude-3.7-sonnet", {"temperature": 0.9, "top_p": 0.95, "top_k": 50}),
            "Technical Writing": ("gpt-4.5", {"temperature": 0.3, "top_p": 0.8, "top_k": 40}),
            "Business Writing": ("gpt-4.5", {"temperature": 0.5, "top_p": 0.85, "top_k": 40}),
            "Educational Content": ("gpt-4.5", {"temperature": 0.4, "top_p": 0.8, "top_k": 40}),
            "Social Media Posts": ("anthropic/claude-3.7-sonnet", {"temperature": 0.8, "top_p": 0.9, "top_k": 50}),
            "Translation": ("gpt-4.5", {"temperature": 0.2, "top_p": 0.7, "top_k": 30}),
            "Summarization": ("gpt-4.5", {"temperature": 0.3, "top_p": 0.8, "top_k": 40}),
            "Question Answering": ("gpt-4.5", {"temperature": 0.4, "top_p": 0.85, "top_k": 40}),
            "General": ("gpt-4.5", {"temperature": 0.6, "top_p": 0.9, "top_k": 50})
        }
        return model_mappings.get(category, ("gpt-4.5", {"temperature": 0.6, "top_p": 0.9, "top_k": 50}))

    async def inlet(self, body: dict, user: dict) -> dict:
        """Processes the first user message and assigns an appropriate LLM model."""
        messages = body.get("messages", [])
        if messages and len(messages) == 1:  # Only process the first message
            user_message = messages[0].get("content", "")
            category = self.detect_category(user_message)
            model, params = self.get_model_and_params(category)

            print(f"Detected category: {category}, Selected model: {model}")

            body["model"] = model
            body.update(params)  # Apply model-specific parameters
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Provides a fallback logic in case `inlet` is not triggered properly."""
        if len(messages) == 1:
            category = self.detect_category(user_message)
            model, params = self.get_model_and_params(category)
            body["model"] = model
            body.update(params)
        return user_message  # Pass-through to the LLM
