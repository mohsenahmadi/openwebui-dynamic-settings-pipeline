import json
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field

# Attempt to import OpenAIChatMessage; fallback if unavailable
try:
    from schemas import OpenAIChatMessage
except ImportError:
    class OpenAIChatMessage:
        pass

class Pipeline:
    class Valves(BaseModel):
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

        CATEGORY_KEYWORDS: dict = Field(
            default={
                "Creative Writing": ["story", "poem", "fiction"],
                "Technical Writing": ["code", "api", "technical"],
                "Question Answering": ["?", "how", "why"]
            },
            description="Keywords for automatic category detection"
        )

        # Additional fields to align with Open WebUI expectations
        pipelines: List[str] = Field(default=[""], description="Target pipeline IDs")
        priority: int = Field(default=0, description="Execution priority")

        # Pydantic v2 configuration
        model_config = {
            "arbitrary_types_allowed": True
        }

        def model_dump(self, *args, **kwargs):
            # Ensure compatibility with serialization
            return self.dict(*args, **kwargs) if hasattr(self, 'dict') else super().model_dump(*args, **kwargs)

    def __init__(self):
        self.name = "AutoTagger Pipeline"
        self.type = "filter"
        # Initialize valves with default values
        self.valves = self.Valves(
            pipelines=[""],  # Connect to all pipelines
            priority=0       # Default priority
        )

    async def on_startup(self):
        print(f"Starting {self.name}")

    async def on_shutdown(self):
        print(f"Shutting down {self.name}")

    async def on_valves_updated(self):
        print(f"Valves updated for {self.name}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if not isinstance(body, dict) or "messages" not in body:
            return body
        
        user_message = self._get_last_user_content(body)
        if user_message:
            category = self._detect_category(user_message)
            body = self._apply_category_settings(body, category)
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if not isinstance(body, dict):
            return body
            
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["processing_pipeline"] = self.name
        return body

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        category = self._detect_category(user_message)
        yield f"Category: {category}\n"
        for chunk in self._generate_response(body):
            yield chunk

    def _get_last_user_content(self, body: dict) -> str:
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def _detect_category(self, text: str) -> str:
        if not text:
            return "DEFAULT"
            
        text_lower = text.lower()
        for category, keywords in self.valves.CATEGORY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "DEFAULT"

    def _apply_category_settings(self, body: dict, category: str) -> dict:
        settings = self.valves.CATEGORY_SETTINGS.get(
            category,
            self.valves.CATEGORY_SETTINGS["DEFAULT"]
        )
        
        for key, value in settings.items():
            if key not in body:
                body[key] = value

        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["detected_category"] = category
        return body

    def _generate_response(self, body: dict) -> Generator:
        yield "Processing complete with parameters:\n"
        params = {
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "max_tokens": body.get("max_tokens")
        }
        yield json.dumps(params, indent=2, ensure_ascii=False)

# Test block for standalone execution
if __name__ == "__main__":
    pipeline = Pipeline()
    
    test_body = {
        "messages": [
            {"role": "user", "content": "Write a story about a dragon"}
        ]
    }
    
    async def test():
        result = await pipeline.inlet(test_body, None)
        print("Inlet result:", result)
        final_result = await pipeline.outlet(result, None)
        print("Outlet result:", final_result)
        
        gen = pipeline.pipe(
            "Write a story",
            "test-model",
            test_body["messages"],
            result
        )
        for chunk in gen:
            print(chunk, end="")

    import asyncio
    asyncio.run(test())
