"""
title: CategoryClassifierPipeline
author: YourName (with help from Grok 3)
version: 1.2
license: MIT
description: A pipeline that classifies the category of the user's first message using microsoft/phi-3-mini-128k-instruct:free model.
requirements: requests
"""

import requests
import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Pipeline:
    class Valves(BaseModel):
        api_url: str = Field(
            default="https://api.openrouter.ai/api/v1/chat/completions",
            description="API URL for the classifier model (e.g., OpenRouter API URL)"
        )
        api_key: str = Field(
            default="",
            description="API Key for the classifier model (e.g., OpenRouter API Key)"
        )

    def __init__(self):
        # Categories for classification
        self.categories = [
            "Creative Writing",
            "Technical Writing",
            "Business Writing",
            "Educational Content",
            "Social Media Posts",
            "Translation",
            "Summarization",
            "Question Answering",
            "Flirting & Seduction Writing",
            "General",
        ]
        # Model for classification
        self.classifier_model = "microsoft/phi-3-mini-128k-instruct:free"
        # Valves will be set in __call__
        self.valves = None

    def __call__(self, data: Dict, config: Optional[Dict] = None) -> Dict:
        """
        Main pipeline function that processes the user message and adds category classification.
        """
        # Load valves (settings)
        self.valves = self.Valves(**config) if config else self.Valves()

        # Check if this is the first user message
        messages = data.get("messages", [])
        if not messages or len(messages) != 1 or messages[0].get("role") != "user":
            return data  # Only process the first user message

        user_message = messages[0].get("content", "")
        if not user_message:
            return data

        # Step 1: Classify the category using the specified model
        category_prompt = (
            f"Classify this request into one of these categories: {', '.join(self.categories)}.\n"
            "Return only the category name in square brackets, e.g., [Summarization].\n\n"
            f"Request: {user_message}"
        )

        try:
            category_response = self._call_classifier_model(category_prompt)
            match = re.match(r"\[(.*?)\]", category_response, re.DOTALL)
            detected_category = match.group(1) if match else "General"
        except Exception as e:
            detected_category = "General"

        # Step 2: Add the category to the message for the default model to process
        # We modify the system message or add a new one to include the category
        system_message = {
            "role": "system",
            "content": f"Category: {detected_category}\nProcess the following user request accordingly."
        }
        messages.insert(0, system_message)
        data["messages"] = messages

        # Add the category to tags
        if "tags" not in data:
            data["tags"] = []
        if detected_category not in data["tags"]:
            data["tags"].append(detected_category)

        # Store in metadata
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["category"] = detected_category

        return data

    def _call_classifier_model(self, prompt: str) -> str:
        """
        Calls the microsoft/phi-3-mini-128k-instruct:free model for category classification.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.api_key}"
        }
        payload = {
            "model": self.classifier_model,
            "messages": [
                {"role": "system", "content": "You are a category classifier."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 50  # We only need a short response for the category
        }

        response = requests.post(self.valves.api_url, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Failed to call classifier model: {response.text}")
