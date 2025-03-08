from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field

class Pipeline:
    class Valves(BaseModel):
        CATEGORY_SETTINGS: dict = Field(
            default={
                "Creative Writing": {"temperature": 0.9},
                "Technical Writing": {"temperature": 0.3},
                "DEFAULT": {"temperature": 0.7}
            },
            description="Model parameters per category"
        )
        
        CATEGORY_KEYWORDS: dict = Field(
            default={
                "Creative Writing": ["story", "poem"],
                "Technical Writing": ["code", "api"]
            },
            description="Keywords for category detection"
        )

    def __init__(self):
        self.name = "CategoryAdapter"
        self.valves = self.Valves()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def on_valves_updated(self):
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # Get last user message
        user_message = next(
            (msg["content"] for msg in reversed(body["messages"]) if msg["role"] == "user"),
            ""
        )
        
        # Detect category
        category = "DEFAULT"
        for cat, keywords in self.valves.CATEGORY_KEYWORDS.items():
            if any(kw in user_message.lower() for kw in keywords):
                category = cat
                break
        
        # Apply category settings
        body.update(self.valves.CATEGORY_SETTINGS.get(category, {}))
        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        yield f"Processed with {self.name}"
