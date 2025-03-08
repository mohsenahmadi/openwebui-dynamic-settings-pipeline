import json
from typing import List, Union, Generator, Iterator

class Pipeline:
    def __init__(self):
        self.name = "AutoTagger Pipeline"
        self.category_settings = {
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
                "top_p": 0.9,
                "max_tokens": None
            }
        }
        self.category_keywords = {
            "Creative Writing": ["story", "poem", "fiction"],
            "Technical Writing": ["code", "api", "technical"],
            "Question Answering": ["?", "how", "why"]
        }

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def on_valves_updated(self):
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        user_message = self._get_last_user_content(body)
        category = self._detect_category(user_message)
        body = self._apply_category_settings(body, category)
        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["processing_pipeline"] = self.name
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        category = self._detect_category(user_message)
        yield f"Category: {category}\n"
        yield from self._generate_response(body)

    def _get_last_user_content(self, body: dict) -> str:
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _detect_category(self, text: str) -> str:
        text_lower = text.lower()
        for category, keywords in self.category_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "DEFAULT"

    def _apply_category_settings(self, body: dict, category: str) -> dict:
        settings = self.category_settings.get(category, self.category_settings["DEFAULT"])
        for k, v in settings.items():
            if body.get(k) is None:
                body[k] = v
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["detected_category"] = category
        return body

    def _generate_response(self, body: dict) -> Generator:
        yield "Processing complete with parameters:\n"
        yield json.dumps({
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "max_tokens": body.get("max_tokens")
        }, indent=2)
