"""
title: CategoryClassifierPipeline
author: YourName (with help from Grok 3)
version: 1.4
license: MIT
description: A pipeline that classifies the category of the user's first message using microsoft/phi-3-mini-128k-instruct:free model.
requirements: requests
"""

import requests
import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

# Helper function to handle Pydantic version compatibility
def pydantic_to_dict(obj):
    try:
        # Pydantic v2
        return obj.model_dump() if obj else {}
    except AttributeError:
        # Pydantic v1
        return obj.dict() if obj else {}

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
        debug: bool = Field(
            default=False,
            description="Enable debug mode to log additional information"
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
        # Initialize valves with default values
        self.valves = self.Valves()

    def __call__(self, data: Dict, config: Optional[Dict] = None) -> Dict:
        """
        Main pipeline function that processes the user message and adds category classification.
        """
        # Load valves (settings)
        if config:
            try:
                self.valves = self.Valves(**config)
            except Exception as e:
                if self.valves.debug:
                    print(f"Error loading config: {str(e)}")
                self.valves = self.Valves()  # Fallback to default
        else:
            self.valves = self.Valves()

        # Debug: Check if valves are loaded correctly
        if self.valves.debug:
            print(f"Config received: {config}")
            print(f"Valves loaded: {pydantic_to_dict(self.valves)}")

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
            if self.valves.debug:
                print(f"Error in category classification: {str(e)}")

        # Step 2: Add the category to the message for the default model to process
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
        if not self.valves:
            raise ValueError("Valves not initialized")

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
