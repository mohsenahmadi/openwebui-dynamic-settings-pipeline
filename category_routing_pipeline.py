"""
title: CategoryRoutingPipeline
author: YourName
date: 2025-03-09
version: 1.0
requirements: 
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import re

class Pipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.name = "Category Routing Pipeline"

    def detect_category(self, message: str) -> str:
        """Detect category using regex with standard library."""
        message = message.lower()
        if re.search(r"story|poem|imagine", message):
            return "Creative Writing"
        elif re.search(r"code|technical|document", message):
            return "Technical Writing"
        elif re.search(r"report|proposal|business", message):
            return "Business Writing"
        elif re.search(r"explain|teach|learn", message):
            return "Educational Content"
        elif re.search(r"post|tweet|share", message):
            return "Social Media Posts"
        elif re.search(r"translate|language|convert", message):
            return "Translation"
        elif re.search(r"summarize|brief|condense", message):
            return "Summarization"
        elif re.search(r"what|how|why", message):
            return "Question Answering"
        else:
            return "General"

    def get_model_and_params(self, category: str) -> tuple:
        """Map category to model and parameters."""
        mappings = {
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
        return mappings.get(category, ("gpt-4.5", {"temperature": 0.6, "top_p": 0.9, "top_k": 50}))

    async def inlet(self, body: dict, user: dict) -> dict:
        """Modify request for the first message only."""
        messages = body.get("messages", [])
        if messages and len(messages) == 1:  # First message in the conversation
            user_message = messages[0]["content"]
            category = self.detect_category(user_message)
            model, params = self.get_model_and_params(category)
            print(f"Detected category: {category}, Selected model: {model}")
            body["model"] = model
            body.update(params)
        # For subsequent messages, rely on Open WebUI’s context to retain the model
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Fallback logic, though inlet should handle most cases."""
        if len(messages) == 1:
            category = self.detect_category(user_message)
            model, params = self.get_model_and_params(category)
            body["model"] = model
            body.update(params)
        return user_message  # Pass through to LLM
