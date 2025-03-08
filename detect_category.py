from typing import List, Optional, Dict
from pydantic import BaseModel, ValidationError
from schemas import OpenAIChatMessage
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        """Configuration for the pipeline filter."""
        pipelines: List[str] = ["*"]  # Apply to all models
        priority: int = 0

    def __init__(self):
        """Initialize the pipeline with type and name."""
        self.type = "filter"
        self.name = "DynamicCategoryLLMAdjusterFilter"
        self.valves = self.Valves()

    async def on_startup(self):
        """Called when the server starts."""
        logger.info(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        """Called when the server stops."""
        logger.info(f"on_shutdown:{__name__}")
        pass

    # Define category settings
    SETTINGS = {
        "Creative Writing": {
            "temperature": 1.0,
            "top_k": 75,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.6,
            "stop": []
        },
        "Technical Writing": {
            "temperature": 0.2,
            "top_k": 15,
            "top_p": 0.5,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1,
            "stop": ["\n\n", "```"]
        },
        "Business Documents": {
            "temperature": 0.6,
            "top_k": 40,
            "top_p": 0.7,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.5,
            "stop": ["Best regards", "Sincerely"]
        },
        "Conversational Responses": {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.8,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.4,
            "stop": ["\n"]
        },
        "Summarization": {
            "temperature": 0.4,
            "top_k": 25,
            "top_p": 0.6,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
            "stop": ["\n", "."]
        }
    }

    def detect_category(self, user_message: str) -> str:
        """Detect the category based on keywords in the user message."""
        message_lower = user_message.lower()
        # Enhanced keyword detection with character count parsing
        if any(keyword in message_lower for keyword in ["write a story", "novel", "poem", "creative"]):
            return "Creative Writing"
        elif any(keyword in message_lower for keyword in ["code", "program", "technical", "document"]):
            return "Technical Writing"
        elif any(keyword in message_lower for keyword in ["email", "report", "proposal", "business"]):
            return "Business Documents"
        elif any(keyword in message_lower for keyword in ["summarize", "summary", "abstract"]):
            return "Summarization"
        else:
            return "Conversational Responses"  # Default for general queries

    def extract_character_limit(self, user_message: str) -> Optional[int]:
        """Extract the character limit from the user message if specified."""
        match = re.search(r"in (\d+) characters?", user_message.lower())
        return int(match.group(1)) if match else None

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Modify the request body to adjust LLM settings dynamically."""
        logger.info(f"inlet:{__name__}")
        logger.info(f"Body: {body}")

        # Safely extract the last user message
        messages = body.get("messages", [])
        user_message = ""
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

        # If no user message found, skip processing
        if not user_message:
            logger.warning("No user message found, skipping category detection.")
            return body

        # Detect category
        category = self.detect_category(user_message)
        logger.info(f"Detected category: {category}")

        # Extract character limit if specified
        character_limit = self.extract_character_limit(user_message)
        if character_limit:
            logger.info(f"Detected character limit: {character_limit}")
            # Adjust max tokens based on character limit (assuming 4 characters per token)
            max_tokens = max(1, character_limit // 4)
        else:
            # Use default max tokens from settings
            max_tokens = self.SETTINGS[category].get("max_tokens", 150)

        # Get settings for the category
        settings = self.SETTINGS[category].copy()
        settings["max_tokens"] = max_tokens

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

        # Append note to the last user message
        note = (
            f"\n\n[Note: Your request was categorized as '{category}'. "
            f"The response will be generated with these settings: "
            f"temperature={settings['temperature']}, top_k={settings['top_k']}, "
            f"top_p={settings['top_p']}, max_tokens={settings['max_tokens']}, "
            f"frequency_penalty={settings['frequency_penalty']}, "
            f"presence_penalty={settings['presence_penalty']}, "
            f"stop={settings['stop']}]"
        )

        # Modify the last user message
        if messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i]["content"] += note
                    break
            else:
                # If no user message exists, append a new one
                messages.append({"role": "user", "content": user_message + note})
            body["messages"] = messages

        logger.info(f"Modified body: {body}")
        return body

if __name__ == "__main__":
    # Example usage for testing (optional)
    class MockBody:
        def __init__(self):
            self.messages = [{"role": "user", "content": "Write a short Story in one hundred characters?"}]

    pipeline = Pipeline()
    body = MockBody().__dict__
    result = pipeline.inlet(body)
    print(result)
