import os
from typing import List, Optional

class Pipeline:
    def __init__(self):
        # Pipeline type
        self.type = "filter"
        self.name = "AutoTagger Pipeline"
        
        # Define valves directly as a dictionary
        self.valves = {
            "pipelines": ["*"],
            "priority": 50
        }
        
        # Store settings and keywords as class attributes for easier access
        self.creative_keywords = ["story", "poem", "fiction", "creative", "write", "narrative"]
        self.technical_keywords = ["code", "api", "technical", "function", "module", "script", "programming"]
        self.question_keywords = ["?", "how", "why", "what", "when", "where", "who"]
        
        self.creative_params = {
            "temperature": 0.9,
            "top_p": 0.95,
            "max_tokens": 2048
        }
        self.technical_params = {
            "temperature": 0.3,
            "top_p": 0.7,
            "max_tokens": 4096
        }
        self.question_params = {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 1024
        }
        self.default_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        }

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    # Add mock dictionary-like access to valves
    def __getattr__(self, name):
        # If something tries to access valves.model_dump, provide a function
        if name == "model_dump":
            return lambda: self.valves

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        
        if not user:
            return body
        
        # Extract the last user message
        user_message = ""
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_message = content
                    break
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    user_message = " ".join(text_parts)
                    break
        
        # Detect category
        category = self._detect_category(user_message)
        
        # Create a copy of body to avoid modifying the original
        modified_body = body.copy()
        
        # Apply parameters based on category
        category_params = self.default_params
        if category == "Creative Writing":
            category_params = self.creative_params
        elif category == "Technical Writing":
            category_params = self.technical_params
        elif category == "Question Answering":
            category_params = self.question_params
        
        # Apply parameters if not already specified
        for key, value in category_params.items():
            if key not in modified_body or modified_body[key] is None:
                modified_body[key] = value
        
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

    def _detect_category(self, text: str) -> str:
        """Detect content category based on keywords"""
        if not text:
            return "DEFAULT"
            
        text_lower = text.lower()
        
        # Check each category
        creative_score = sum(1 for kw in self.creative_keywords if kw.lower() in text_lower)
        technical_score = sum(1 for kw in self.technical_keywords if kw.lower() in text_lower)
        question_score = sum(1 for kw in self.question_keywords if kw.lower() in text_lower)
        
        # Determine category with highest score
        scores = {
            "Creative Writing": creative_score,
            "Technical Writing": technical_score,
            "Question Answering": question_score
        }
        
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "DEFAULT"
