from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage  # (optional utilities for message format)
from utils.pipelines.main import get_last_user_message

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]    # Apply to all model pipelines (all chats)&#8203;:contentReference[oaicite:6]{index=6}
        priority: int = 0              # Priority of this filter (0 = high priority)
        # (You could add more config valves here if needed, but not required)

    def __init__(self):
        self.type = "filter"
        self.name = "AutoParamAdjust"
        self.valves = self.Valves()    # Initialize valves as defined above

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This runs BEFORE sending the request to the LLM&#8203;:contentReference[oaicite:7]{index=7}
        messages = body.get("messages", [])
        user_msg = get_last_user_message(messages)
        if user_msg is not None:
            prompt = user_msg.get("content", "").lower()

            # --- Categorize the prompt ---
            category = "general"
            if any(keyword in prompt for keyword in ["story", "poem", "imagine"]):
                category = "creative"
            elif any(keyword in prompt for keyword in ["explain", "definition", "what is", "how to"]):
                category = "technical"
            elif any(keyword in prompt for keyword in ["code", "function", "algorithm"]):
                category = "code"
            # (You can expand these rules or use an LLM call for classification)

            # --- Adjust parameters based on category ---
            if category == "creative":
                body["temperature"] = 0.9   # more randomness for creative writing
                body["top_p"] = 0.95        # use top-p sampling for more diverse output
                body["max_tokens"] = 1024   # allow longer output for stories/poems
            elif category == "technical":
                body["temperature"] = 0.2   # low randomness for factual accuracy
                body["top_p"] = 0.8         # focus on likely tokens
                body["max_tokens"] = 512    # moderate length for explanations
            elif category == "code":
                body["temperature"] = 0.1   # very deterministic for code
                body["top_p"] = 0.5         # reduce creativity
                body["max_tokens"] = 256    # code answers usually shorter
            else:
                # general/default – you can either leave defaults or set a baseline
                body["temperature"] = 0.7
                body["top_p"] = 0.9
                body["max_tokens"] = 768

            # (Optionally, adjust other parameters like presence_penalty, etc., here)

        # Return the modified request body
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Not needed for this use-case, but must be defined. We’ll just return the output unchanged.
        return body
