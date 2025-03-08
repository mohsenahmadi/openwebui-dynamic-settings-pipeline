import os
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import time

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = ["*"]
        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 50
        
        # Our custom settings
        creative_keywords: List[str] = ["story", "poem", "fiction", "creative", "write", "narrative"]
        technical_keywords: List[str] = ["code", "api", "technical", "function", "module", "script", "programming"]
        question_keywords: List[str] = ["?", "how", "why", "what", "when", "where", "who"]

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "autotagger_filter_pipeline"
        self.name = "AutoTagger Pipeline"
        self.valves = self.Valves()
        
        # Store category parameters separately
        self.category_params = {
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
        print(user)
        
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
        
        # Apply parameters for the detected category
        params = self.category_params.get(category, self.category_params["DEFAULT"])
        for key, value in params.items():
            if key not in modified_body or modified_body[key] is None:
                modified_body[key] = value
        
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
