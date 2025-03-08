import os
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        pipelines: List[str] = ["*"]
        # Assign a priority level to the filter pipeline.
        priority: int = 50
        # Category-specific settings
        # Keywords for automatic category detection
        creative_keywords: List[str] = ["story", "poem", "fiction", "creative", "write", "narrative"]
        technical_keywords: List[str] = ["code", "api", "technical", "function", "module", "script", "programming"]
        question_keywords: List[str] = ["?", "how", "why", "what", "when", "where", "who"]
        # Model parameters for each category
        creative_temperature: float = 0.9
        creative_top_p: float = 0.95
        creative_max_tokens: int = 2048
        technical_temperature: float = 0.3
        technical_top_p: float = 0.7
        technical_max_tokens: int = 4096
        question_temperature: float = 0.7
        question_top_p: float = 0.8
        question_max_tokens: int = 1024
        default_temperature: float = 0.7
        default_top_p: float = 0.9
        default_max_tokens: int = 2048

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        self.type = "filter"
        self.name = "AutoTagger Pipeline"
        self.valves = self.Valves()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        print(body)
        
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
        if category == "Creative Writing":
            if "temperature" not in modified_body:
                modified_body["temperature"] = self.valves.creative_temperature
            if "top_p" not in modified_body:
                modified_body["top_p"] = self.valves.creative_top_p
            if "max_tokens" not in modified_body:
                modified_body["max_tokens"] = self.valves.creative_max_tokens
        elif category == "Technical Writing":
            if "temperature" not in modified_body:
                modified_body["temperature"] = self.valves.technical_temperature
            if "top_p" not in modified_body:
                modified_body["top_p"] = self.valves.technical_top_p
            if "max_tokens" not in modified_body:
                modified_body["max_tokens"] = self.valves.technical_max_tokens
        elif category == "Question Answering":
            if "temperature" not in modified_body:
                modified_body["temperature"] = self.valves.question_temperature
            if "top_p" not in modified_body:
                modified_body["top_p"] = self.valves.question_top_p
            if "max_tokens" not in modified_body:
                modified_body["max_tokens"] = self.valves.question_max_tokens
        else:  # DEFAULT
            if "temperature" not in modified_body:
                modified_body["temperature"] = self.valves.default_temperature
            if "top_p" not in modified_body:
                modified_body["top_p"] = self.valves.default_top_p
            if "max_tokens" not in modified_body:
                modified_body["max_tokens"] = self.valves.default_max_tokens
        
        # Add metadata
        if "metadata" not in modified_body:
            modified_body["metadata"] = {}
        modified_body["metadata"]["detected_category"] = category
        
        return modified_body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This function is called after the OpenAI API responds.
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
        creative_score = sum(1 for kw in self.valves.creative_keywords if kw.lower() in text_lower)
        technical_score = sum(1 for kw in self.valves.technical_keywords if kw.lower() in text_lower)
        question_score = sum(1 for kw in self.valves.question_keywords if kw.lower() in text_lower)
        
        # Determine category with highest score
        scores = {
            "Creative Writing": creative_score,
            "Technical Writing": technical_score,
            "Question Answering": question_score
        }
        
        if max(scores.values()) > 0:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "DEFAULT"
