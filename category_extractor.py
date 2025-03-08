import os
import json
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel

class CategorySettings(BaseModel):
    temperature: float
    top_p: float
    max_tokens: int = None

class Valves(BaseModel):
    CATEGORY_SETTINGS: dict = {
        "Creative Writing": CategorySettings(
            temperature=0.9, top_p=0.95, max_tokens=2048
        ),
        "Technical Writing": CategorySettings(
            temperature=0.3, top_p=0.7, max_tokens=4096
        ),
        "DEFAULT": CategorySettings(
            temperature=0.7, top_p=0.9, max_tokens=None
        )
    }
    CATEGORY_KEYWORDS: dict = {
        "Creative Writing": ["story", "poem", "fiction"],
        "Technical Writing": ["code", "api", "technical"],
        "Question Answering": ["?", "how", "why"]
    }

class Pipeline:
    def __init__(self):
        self.name = "AutoTagger Pipeline"
        self.valves = Valves()

    async def on_startup(self):
        # Initialize pipeline components
        pass

    async def on_shutdown(self):
        # Clean up resources
        pass

    async def on_valves_updated(self):
        # Handle configuration changes
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # Detect category and apply parameters before API request
        user_message = self._get_last_user_content(body)
        category = self._detect_category(user_message)
        body = self._apply_category_settings(body, category)
        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        # Add metadata to final response
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["processing_pipeline"] = self.name
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Stream processing with category context
        category = self._detect_category(user_message)
        yield f"Category: {category}\n"
        yield from self._generate_response(body)

    def _get_last_user_content(self, body: dict) -> str:
        # Extract last user message from chat history
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _detect_category(self, text: str) -> str:
        # Simple keyword-based categorization
        text_lower = text.lower()
        for category, keywords in self.valves.CATEGORY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "DEFAULT"

    def _apply_category_settings(self, body: dict, category: str) -> dict:
        # Apply model parameters based on detected category
        settings = self.valves.CATEGORY_SETTINGS.get(
            category,
            self.valves.CATEGORY_SETTINGS["DEFAULT"]
        )
        
        # Update model parameters
        body.update({k: v for k, v in settings.dict().items() if k not in body})
        
        # Add category metadata
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["detected_category"] = category
        
        return body

    def _generate_response(self, body: dict) -> Generator:
        # Simulated response generation
        yield "Processing complete with parameters:\n"
        yield json.dumps({
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "max_tokens": body.get("max_tokens")
        }, indent=2)
