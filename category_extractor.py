import json
from typing import List, Union, Generator, Iterator, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel, Field

class Pipeline:
    class Valves(BaseModel):
        # Category configuration with model parameters
        CATEGORY_SETTINGS: dict = Field(
            default={
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
                "DEFAULT": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            description="Model parameters per content category"
        )
        
        # Category detection settings
        CATEGORY_KEYWORDS: dict = Field(
            default={
                "Creative Writing": ["story", "poem", "fiction"],
                "Technical Writing": ["code", "api", "technical"],
                "Question Answering": ["?", "how", "why"]
            },
            description="Keywords for automatic category detection"
        )

    def __init__(self):
        self.name = "AutoTagger Pipeline"
        
        # برای پشتیبانی از Pydantic v1 و v2
        try:
            self.valves = self.Valves().model_dump()  # Pydantic v2
        except AttributeError:
            self.valves = self.Valves().dict()  # Pydantic v1

    async def on_startup(self):
        print(f"{self.name} started.")

    async def on_shutdown(self):
        print(f"{self.name} shutting down.")

    async def on_valves_updated(self):
        print(f"{self.name} valves updated.")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        user_message = self._get_last_user_content(body)
        category = self._detect_category(user_message)
        body = self._apply_category_settings(body, category)
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["processing_pipeline"] = self.name
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        category = self._detect_category(user_message)
        body = self._apply_category_settings(body, category)
        yield f"Category: {category}\n"
        yield from self._generate_response(body)

    def _get_last_user_content(self, body: dict) -> str:
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _detect_category(self, text: str) -> str:
        text_lower = text.lower()
        for category, keywords in self.valves["CATEGORY_KEYWORDS"].items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "DEFAULT"

    def _apply_category_settings(self, body: dict, category: str) -> dict:
        settings = self.valves["CATEGORY_SETTINGS"].get(category, self.valves["CATEGORY_SETTINGS"]["DEFAULT"])
        for k, v in settings.items():
            body.setdefault(k, v)
        
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["detected_category"] = category
        
        return body

    def _generate_response(self, body: dict):
        yield "Processing complete with parameters:\n"
        yield json.dumps({
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "max_tokens": body.get("max_tokens")
        }, indent=2)
