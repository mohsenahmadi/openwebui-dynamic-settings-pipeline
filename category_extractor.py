import os
import json
from typing import List, Optional, Dict, Any

class Pipeline:
    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        self.type = "filter"
        self.name = "AutoTagger Pipeline"
        
        # Define core valves structure expected by the framework
        self.valves = self._create_valves_object()
    
    def _create_valves_object(self):
        """Create a valves object compatible with the core framework"""
        # This is a dictionary that mimics what Pydantic would typically provide
        return {
            "pipelines": ["*"],
            "priority": 50,
            
            # Store our category settings directly in the dictionary
            "creative_writing_params": {
                "temperature": 0.9,
                "top_p": 0.95,
                "max_tokens": 2048
            },
            "technical_writing_params": {
                "temperature": 0.3,
                "top_p": 0.7,
                "max_tokens": 4096
            },
            "question_answering_params": {
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 1024
            },
            "default_params": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2048
            },
            
            # Keywords for detection
            "creative_keywords": ["story", "poem", "fiction", "creative", "write", "narrative"],
            "technical_keywords": ["code", "api", "technical", "function", "module", "script", "programming"],
            "question_keywords": ["?", "how", "why", "what", "when", "where", "who"],
            
            # Add model_dump method to make it compatible
            "model_dump": lambda: self.valves
        }

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        
        if not user:
            return body
            
        # Extract the last user message
        user_message = self._get_last_user_content(body)
        
        # Detect category
        category = self._detect_category(user_message)
        
        # Apply category-specific parameters
        modified_body = self._apply_category_params(body, category)
        
        # Add metadata
        if "metadata" not in modified_body:
            modified_body["metadata"] = {}
        modified_body["metadata"]["detected_category"] = category
        
        return modified_body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if not user:
            return body
            
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["processing_pipeline"] = self.name
        return body
    
    def _get_last_user_content(self, body: dict) -> str:
        """Extract the last user message content from the request body"""
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    return " ".join(text_parts)
        return ""

    def _detect_category(self, text: str) -> str:
        """Detect content category based on keywords"""
        if not text:
            return "DEFAULT"
            
        text_lower = text.lower()
        
        # Check each category
        creative_score = sum(1 for kw in self.valves["creative_keywords"] if kw.lower() in text_lower)
        technical_score = sum(1 for kw in self.valves["technical_keywords"] if kw.lower() in text_lower)
        question_score = sum(1 for kw in self.valves["question_keywords"] if kw.lower() in text_lower)
        
        # Determine category with highest score
        scores = {
            "Creative Writing": creative_score,
            "Technical Writing": technical_score,
            "Question Answering": question_score
        }
        
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "DEFAULT"

    def _apply_category_params(self, body: dict, category: str) -> dict:
        """Apply model parameters based on detected category"""
        # Get parameters for the detected category
        params = self.valves["default_params"]
        if category == "Creative Writing":
            params = self.valves["creative_writing_params"]
        elif category == "Technical Writing":
            params = self.valves["technical_writing_params"]
        elif category == "Question Answering":
            params = self.valves["question_answering_params"]
        
        # Create a copy of body to avoid modifying the original
        modified_body = body.copy()
        
        # Apply parameters if not already specified
        for key, value in params.items():
            if key not in modified_body or modified_body[key] is None:
                modified_body[key] = value
        
        return modified_body
