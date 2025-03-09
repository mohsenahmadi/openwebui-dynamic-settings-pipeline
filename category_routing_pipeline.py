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
        # Optional: Configure category keywords or model mappings here
        pass

    def __init__(self):
        self.name = "Category Routing Pipeline"
        # Track the selected model and parameters per chat session
        self.session_state = {}

    # Category detection logic
    def detect_category(self, message: str) -> str:
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

    # Model and parameter mapping
    def get_model_and_params(self, category: str) -> tuple:
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
        # Extract chat ID to track session (assumes Open WebUI provides a unique identifier)
        chat_id = body.get("chat_id", user.get("id", "default"))

        # Check if this is the first request in the session
        if chat_id not in self.session_state:
            user_message = body.get("messages", [])[-1]["content"] if body.get("messages") else ""
            category = self.detect_category(user_message)
            model, params = self.get_model_and_params(category)
            self.session_state[chat_id] = {"model": model, "params": params}
            print(f"Detected category: {category}, Selected model: {model}")

        # Apply the selected model and parameters
        body["model"] = self.session_state[chat_id]["model"]
        body.update(self.session_state[chat_id]["params"])
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # Fallback if inlet isn’t triggered; not typically needed with Arena Model
        chat_id = body.get("chat_id", "default")
        if chat_id not in self.session_state:
            category = self.detect_category(user_message)
            model, params = self.get_model_and_params(category)
            self.session_state[chat_id] = {"model": model, "params": params}
        return user_message  # Let the downstream LLM handle the response
