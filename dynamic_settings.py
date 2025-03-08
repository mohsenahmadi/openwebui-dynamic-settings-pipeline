from typing import Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage  # Open WebUI message schema, if needed

class Pipeline:
    class Valves(BaseModel):
        pipelines: list[str] = ["*"]       # Apply to all model pipelines
        priority: int = 0                 # High priority (execute early)
        # (Optional) We could add configurable thresholds or mappings here as valves

    def __init__(self):
        self.type = "filter"
        self.name = "DynamicParamTuner"
        self.valves = self.Valves()  # using defaults above

        # Define mapping from category to parameter settings
        self.param_presets = {
            "code":    {"temperature": 0.0, "top_p": 0.15, "top_k": 1,   "repetition_penalty": 1.0,  "max_tokens": 512},
            "story":   {"temperature": 0.9, "top_p": 0.95, "top_k": 0,   "repetition_penalty": 1.1,  "max_tokens": 1024},
            "business":{"temperature": 0.4, "top_p": 0.5,  "top_k": 40,  "repetition_penalty": 1.1,  "max_tokens": 512},
            "science": {"temperature": 0.2, "top_p": 0.3,  "top_k": 40,  "repetition_penalty": 1.05, "max_tokens": 512},
            "casual":  {"temperature": 0.6, "top_p": 0.7,  "top_k": 40,  "repetition_penalty": 1.15, "max_tokens": 256},
            "default": {"temperature": 0.7, "top_p": 0.9,  "top_k": 40,  "repetition_penalty": 1.1,  "max_tokens": 512}
        }
        # Note: max_tokens here is an example; adjust as needed or omit if handled elsewhere.

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Analyze the incoming user prompt to determine category
        messages = body.get("messages", [])
        # Get the latest user message content
        user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "").lower()
                break

        category = "default"
        if user_msg:
            if "code" in user_msg or "function" in user_msg or "```" in user_msg:
                category = "code"
            elif "story" in user_msg or "once upon a time" in user_msg or "novel" in user_msg:
                category = "story"
            elif "report" in user_msg or "proposal" in user_msg or "business" in user_msg or "dear sir" in user_msg:
                category = "business"
            elif "research" in user_msg or "study" in user_msg or "scientific" in user_msg or "paper" in user_msg:
                category = "science"
            elif "chat" in user_msg or "conversation" in user_msg or "casual" in user_msg:
                category = "casual"
            # (Additional rules can be added for other categories)

        # Fetch the preset for the detected category (or default if not found)
        params = self.param_presets.get(category, self.param_presets["default"])
        # Apply each parameter to the request body
        for param, value in params.items():
            body[param] = value

        body["__category"] = category  # stash the category, maybe for use in outlet
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # After model generates a response, add an annotation about category & settings
        category = body.get("__category", "default")
        used_params = self.param_presets.get(category, {})
        # Construct a brief note
        settings_summary = ", ".join(f"{k}={v}" for k,v in used_params.items() if k != "max_tokens")
        note = f"*[Detected **{category.capitalize()}** request – used {settings_summary}]*\n\n"

        # Prepend the note to the assistant's last message
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                msg["content"] = note + msg["content"]
                break
        # (If no assistant message found for some reason, we do nothing)
        return body
