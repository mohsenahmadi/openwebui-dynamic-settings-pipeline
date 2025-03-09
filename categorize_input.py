# categorize_input.py
# Filter Pipeline to categorize user input and tag with model/settings recommendations

from typing import List, Optional, Dict
from pydantic import BaseModel

CATEGORIES = {
    "Creative Writing": ("anthropic/claude-3.7-sonnet", {"temperature": 0.9, "top_p": 0.95, "top_k": 50}),
    "Technical Writing": ("gpt-4.5", {"temperature": 0.3, "top_p": 0.8, "top_k": 40}),
    "Business Writing": ("gpt-4.5", {"temperature": 0.5, "top_p": 0.85, "top_k": 40}),
    "Educational Content": ("gpt-4.5", {"temperature": 0.4, "top_p": 0.8, "top_k": 40}),
    "Social Media Posts": ("anthropic/claude-3.7-sonnet", {"temperature": 0.8, "top_p": 0.9, "top_k": 50}),
    "Translation": ("gpt-4.5", {"temperature": 0.2, "top_p": 0.7, "top_k": 30}),
    "Summarization": ("gpt-4.5", {"temperature": 0.3, "top_p": 0.8, "top_k": 40}),
    "Question Answering": ("gpt-4.5", {"temperature": 0.4, "top_p": 0.85, "top_k": 40}),
    "General": ("gpt-4.5", {"temperature": 0.6, "top_p": 0.9, "top_k": 50}),
}

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]  # Apply to all models
        priority: int = 0  # Default priority

    def __init__(self):
        self.type = "filter"
        self.name = "Categorize Input Pipeline"
        self.valves = self.Valves()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Extract the last user message
        messages = body.get("messages", [])
        if not messages:
            return body
        
        question = messages[-1].get("content", "").lower()

        # Keyword-based categorization
        if "write a story" in question or "creative" in question:
            category = "Creative Writing"
        elif "technical" in question or "code" in question:
            category = "Technical Writing"
        elif "business" in question or "email" in question:
            category = "Business Writing"
        elif "educat" in question or "teach" in question:
            category = "Educational Content"
        elif "social media" in question or "post" in question:
            category = "Social Media Posts"
        elif "translate" in question:
            category = "Translation"
        elif "summarize" in question:
            category = "Summarization"
        elif "what" in question or "how" in question:
            category = "Question Answering"
        else:
            category = "General"

        model, settings = CATEGORIES[category]
        
        # Add tags to the body
        if "tags" not in body:
            body["tags"] = {}
        body["tags"].update({
            "category": category,
            "model": model,
            "settings": settings,
            "original_input": question
        })

        return body
