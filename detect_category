from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]  # Apply to all models
        priority: int = 0

    def __init__(self):
        self.type = "filter"
        self.name = "DynamicCategoryLLMAdjusterFilter"
        self.valves = self.Valves()

    # Same SETTINGS and detect_category as before
    SETTINGS = { ... }  # Copy from previous code
    def detect_category(self, user_message: str) -> str:
        # Copy from previous code
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        user_message = body.get("messages", [])[-1]["content"] if body.get("messages") else ""
        category = self.detect_category(user_message)
        settings = self.SETTINGS[category]

        # Update body with settings
        body.update({
            "temperature": settings["temperature"],
            "top_k": settings["top_k"],
            "top_p": settings["top_p"],
            "max_tokens": settings["max_tokens"],
            "frequency_penalty": settings["frequency_penalty"],
            "presence_penalty": settings["presence_penalty"],
            "stop": settings["stop"]
        })

        # Append note to the last message
        note = f"\n\n[Note: Your request was categorized as '{category}'...]"  # Full note as before
        body["messages"][-1]["content"] += note

        print(f"Modified body: {body}")
        return body
