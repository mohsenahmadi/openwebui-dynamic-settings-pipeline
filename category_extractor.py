from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field

class Pipeline:
    class Valves(BaseModel):
        CATEGORY_PROFILES: dict = Field(
            default={
                "Creative Writing": {"temperature": 0.9},
                "Technical Writing": {"temperature": 0.3},
                "DEFAULT": {"temperature": 0.7}
            },
            description="Model parameters per category"
        )
        
        KEYWORD_MAPPING: dict = Field(
            default={
                "Creative Writing": ["story", "poem"],
                "Technical Writing": ["code", "api"]
            },
            description="Category detection keywords"
        )

    def __init__(self):
        self.name = "CategoryAdapter"
        self.valves = self.Valves(**self.Valves.__fields__['CATEGORY_PROFILES'].default,
                                  **self.Valves.__fields__['KEYWORD_MAPPING'].default)

    async def inlet(self, body: dict, user: dict) -> dict:
        # Get last user message
        last_msg = next((m for m in reversed(body.get("messages", [])) if m["role"] == "user", None)
        user_content = last_msg.get("content", "") if last_msg else ""

        # Detect category
        detected_category = "DEFAULT"
        for category, keywords in self.valves.KEYWORD_MAPPING.items():
            if any(kw in user_content.lower() for kw in keywords):
                detected_category = category
                break

        # Apply settings
        profile = self.valves.CATEGORY_PROFILES.get(detected_category, {})
        return {**body, **profile}

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        yield f"Processing completed with category: {body.get('detected_category', 'DEFAULT')}"

    # بقیه متدها بدون تغییر
    async def on_startup(self): ...
    async def on_shutdown(self): ...
    async def on_valves_updated(self): ...
    async def outlet(self, body: dict, user: dict) -> dict: return body
