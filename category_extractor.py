import os
import json
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = ["*"]
        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 50
        # Category configuration with model parameters
        creative_writing_params: dict = {
            "temperature": 0.9,
            "top_p": 0.95,
            "max_tokens": 2048
        }
        technical_writing_params: dict = {
            "temperature": 0.3,
            "top_p": 0.7,
            "max_tokens": 4096
        }
        question_answering_params: dict = {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 1024
        }
        default_params: dict = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        }
        # Keywords for category detection
        creative_keywords: List[str] = ["story", "poem", "fiction", "creative", "write", "narrative"]
        technical_keywords: List[str] = ["code", "api", "technical", "function", "module", "script", "programming"]
        question_keywords: List[str] = ["?", "how", "why", "what", "when", "where", "who"]

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename
        # self.id = "autotagger_filter_pipeline"
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
        
        # Apply category-specific parameters
        body = self._apply_category_params(body, category)
        
        # Add metadata
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["detected_category"] = category
        
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Add metadata to final response
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

    def _apply_category_params(self, body: dict, category: str) -> dict:
        """Apply model parameters based on detected category"""
        # Get parameters for the detected category
        params = self.valves.default_params
        if category == "Creative Writing":
            params = self.valves.creative_writing_params
        elif category == "Technical Writing":
            params = self.valves.technical_writing_params
        elif category == "Question Answering":
            params = self.valves.question_answering_params
        
        # Create a copy of body to avoid modifying the original
        modified_body = body.copy()
        
        # Apply parameters if not already specified
        for key, value in params.items():
            if key not in modified_body or modified_body[key] is None:
                modified_body[key] = value
        
        return modified_body
