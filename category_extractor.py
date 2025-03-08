import os
import json
from typing import List, Union, Generator, Iterator, Optional, Dict, Any
from schemas import OpenAIChatMessage
from pydantic import BaseModel, Field

class Pipeline:
    class Valves(BaseModel):
        # Category configuration with model parameters
        CATEGORY_SETTINGS: Dict[str, Dict[str, Any]] = {
            "Creative Writing": {
                "temperature": 0.9,
                "top_p": 0.95,
                "max_tokens": 2048
            },
            "Technical Writing": {
                "temperature": 0.3,
                "top_p": 0.7,
                "max_tokens": 4096
            },
            "Question Answering": {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 1024
            },
            "DEFAULT": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }
        
        # Category detection settings
        CATEGORY_KEYWORDS: Dict[str, List[str]] = {
            "Creative Writing": ["story", "poem", "fiction", "creative", "write", "narrative"],
            "Technical Writing": ["code", "api", "technical", "function", "module", "script", "programming"],
            "Question Answering": ["?", "how", "why", "what", "when", "where", "who"]
        }
        
        # Pipeline settings
        pipelines: List[str] = ["*"]
        priority: int = 50
        
        class Config:
            # Configure for Pydantic v1 compatibility
            validate_assignment = True

    def __init__(self):
        self.type = "filter"  # Required for filter pipelines
        self.name = "AutoTagger Pipeline"
        self.valves = self.Valves()

    async def on_startup(self):
        # Initialize pipeline components
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # Clean up resources
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # Handle configuration changes
        print(f"on_valves_updated:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Detect category and apply parameters before API request
        print(f"pipe:{__name__}")
        
        if not user:
            return body
            
        user_message = self._get_last_user_content(body)
        category = self._detect_category(user_message)
        body = self._apply_category_settings(body, category)
        
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Add metadata to final response
        if not user:
            return body
            
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["processing_pipeline"] = self.name
        return body

    def _get_last_user_content(self, body: dict) -> str:
        # Extract last user message from chat history
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                # Handle both string content and complex content structures
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Extract text from content parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    return " ".join(text_parts)
        return ""

    def _detect_category(self, text: str) -> str:
        # Simple keyword-based categorization
        if not text:
            return "DEFAULT"
            
        text_lower = text.lower()
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.valves.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or DEFAULT if no matches
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return "DEFAULT"

    def _apply_category_settings(self, body: dict, category: str) -> dict:
        # Apply model parameters based on detected category
        settings = self.valves.CATEGORY_SETTINGS.get(
            category,
            self.valves.CATEGORY_SETTINGS["DEFAULT"]
        )
        
        # Create a copy of body to avoid modifying the original
        modified_body = body.copy()
        
        # Update model parameters if not already specified
        for key, value in settings.items():
            if key not in modified_body or modified_body[key] is None:
                modified_body[key] = value
        
        # Add category metadata
        if "metadata" not in modified_body:
            modified_body["metadata"] = {}
        modified_body["metadata"]["detected_category"] = category
        
        return modified_body
