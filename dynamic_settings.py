"""
title: Dynamic Settings Filter Pipeline
author: [Your Name]
date: 2025-03-08
version: 1.0
license: MIT
description: A filter pipeline that dynamically adjusts OpenAI API settings based on the task type inferred from the user's prompt.
requirements: None
"""

from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage


class Pipeline:
    class Valves(BaseModel):
        # Connect this filter to all pipelines by default (use "*" for all models)
        pipelines: List[str] = ["*"]

        # Set priority (lower number = higher priority)
        priority: int = 0

    def __init__(self):
        # Define this as a filter pipeline for Open WebUI
        self.type = "filter"

        # Name of the pipeline
        self.name = "DynamicSettings"

        # Configure the valves (target pipelines and priority)
        self.valves = self.Valves(**{"pipelines": ["*"]})

    async def on_startup(self):
        # Called when the server starts
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # Called when the server stops
        print(f"on_shutdown:{__name__}")
        pass

    def classify_task(self, prompt: str) -> str:
        """Classify the task type based on keywords in the prompt."""
        prompt = prompt.lower()
        if any(keyword in prompt for keyword in ["story", "imagine", "create", "poem"]):
            return "creative"
        elif any(keyword in prompt for keyword in ["what is", "explain", "define", "history"]):
            return "factual"
        elif any(keyword in prompt for keyword in ["code", "program", "function", "debug"]):
            return "technical"
        return "default"

    # Settings map for different task types
    SETTINGS_MAP = {
        "creative": {
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "max_tokens": 1000
        },
        "factual": {
            "temperature": 0.2,
            "top_k": 10,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 100
        },
        "technical": {
            "temperature": 0.3,
            "top_k": 20,
            "top_p": 0.85,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 300
        },
        "default": {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.2,
            "max_tokens": 400
        }
    }

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Filter applied to form data before sending to the OpenAI API."""
        print(f"inlet:{__name__}")

        # Extract the prompt from the last message in the body
        messages = body.get("messages", [])
        if messages:
            prompt = messages[-1].get("content", "")
        else:
            prompt = ""

        # Classify the task and get the corresponding settings
        task = self.classify_task(prompt)
        settings = self.SETTINGS_MAP.get(task, self.SETTINGS_MAP["default"])

        # Apply the settings to the body
        body["temperature"] = settings["temperature"]
        body["top_k"] = settings["top_k"]
        body["top_p"] = settings["top_p"]
        body["frequency_penalty"] = settings["frequency_penalty"]
        body["presence_penalty"] = settings["presence_penalty"]
        body["max_tokens"] = settings["max_tokens"]

        print(f"Task classified as: {task}")
        print(f"Applied settings: {settings}")
        print(body)

        return body
