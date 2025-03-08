"""
title: Dynamic Settings Manifold Pipeline
author: D!Mo
date: 2025-03-08
version: 1.1
license: MIT
description: A manifold pipeline that dynamically adjusts OpenAI API settings and appends the applied settings map to the response.
requirements: None
"""

from typing import List, Optional, Union, Generator, Iterator
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0

    def __init__(self):
        # Change type to manifold to handle both input and output
        self.type = "manifold"
        self.name = "DynamicSettings"
        self.valves = self.Valves(**{"pipelines": ["*"]})
        # Store the task type for use in outlet
        self.current_task = "default"

    async def on_startup(self):
        logger.info(f"Pipeline started: {__name__}")

    async def on_shutdown(self):
        logger.info(f"Pipeline stopped: {__name__}")

    def classify_task(self, prompt: str) -> str:
        prompt = prompt.lower()
        if any(keyword in prompt for keyword in ["story", "imagine", "create", "poem"]):
            return "creative"
        elif any(keyword in prompt for keyword in ["what is", "explain", "define", "history"]):
            return "factual"
        elif any(keyword in prompt for keyword in ["code", "program", "function", "debug"]):
            return "technical"
        return "default"

    SETTINGS_MAP = {
        "creative": {"temperature": 0.9, "top_k": 50, "top_p": 0.95, "frequency_penalty": 0.5, "presence_penalty": 0.5, "max_tokens": 1000},
        "factual": {"temperature": 0.2, "top_k": 10, "top_p": 0.9, "frequency_penalty": 0.0, "presence_penalty": 0.0, "max_tokens": 100},
        "technical": {"temperature": 0.3, "top_k": 20, "top_p": 0.85, "frequency_penalty": 0.0, "presence_penalty": 0.0, "max_tokens": 300},
        "default": {"temperature": 0.7, "top_k": 40, "top_p": 0.9, "frequency_penalty": 0.5, "presence_penalty": 0.2, "max_tokens": 400}
    }

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        logger.info(f"Processing request in {__name__}")

        # Extract prompt and classify task
        messages = body.get("messages", [])
        prompt = messages[-1].get("content", "") if messages else ""
        task = self.classify_task(prompt)
        self.current_task = task  # Store task for use in outlet
        settings = self.SETTINGS_MAP.get(task, self.SETTINGS_MAP["default"])

        # Apply settings
        body["temperature"] = settings["temperature"]
        body["top_k"] = settings["top_k"]
        body["top_p"] = settings["top_p"]
        body["frequency_penalty"] = settings["frequency_penalty"]
        body["presence_penalty"] = settings["presence_penalty"]
        body["max_tokens"] = settings["max_tokens"]

        logger.info(f"Task: {task}, User: {user.get('name', 'Unknown') if user else 'Unknown'}")
        logger.info(f"Applied settings: {settings}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> Union[str, Generator, Iterator]:
        logger.info(f"Processing response in {__name__}")

        # If body is a string (direct response), append the task type
        if isinstance(body, str):
            return f"{body}\n\n({self.current_task})"

        # If body is a generator/iterator (streaming response), wrap it to append the task type
        if isinstance(body, (Generator, Iterator)):
            async def wrapped_generator():
                async for chunk in body:
                    yield chunk
                yield f"\n\n({self.current_task})"
            return wrapped_generator()

        # If body is a dict (structured response), append to the content
        if isinstance(body, dict) and "content" in body:
            body["content"] = f"{body['content']}\n\n({self.current_task})"
            return body

        # Fallback: return unchanged if format is unrecognized
        logger.warning("Unrecognized response format, returning unchanged.")
        return body
