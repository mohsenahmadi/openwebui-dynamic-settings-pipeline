from typing import List, Optional, Dict
from pydantic import BaseModel
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

    # Define category settings with only essential parameters
    SETTINGS = {
        "Creative Writing": {
            "temperature": 1.0,
            "max_tokens": 750,
            "stop": []
        },
        "Technical Writing": {
            "temperature": 0.2,
            "max_tokens": 300,
            "stop": ["\n\n", "```"]
        },
        "Business Documents": {
            "temperature": 0.6,
            "max_tokens": 200,
            "stop": ["Best regards", "Sincerely"]
        },
        "Conversational Responses": {
            "temperature": 0.7,
            "max_tokens": 150,
            "stop": ["\n"]
        },
        "Summarization": {
            "temperature": 0.4,
            "max_tokens": 100,
            "stop": ["\n", "."]
        }
    }

    def detect_category(self, user_message: str) -> str:
        """Detect the category based on keywords in the user message."""
        message_lower = user_message.lower()
        if any(keyword in message_lower for keyword in ["write a story", "novel", "poem", "creative"]):
            return "Creative Writing"
        elif any(keyword in message_lower for keyword in ["code", "program", "technical", "document"]):
            return "Technical Writing"
        elif any(keyword in message_lower for keyword in ["email", "report", "proposal", "business"]):
            return "Business Documents"
        elif any(keyword in message_lower for keyword in ["summarize", "summary", "abstract"]):
            return "Summarization"
        else:
            return "Conversational Responses"

    def extract_character_limit(self, user_message: str) -> Optional[int]:
        """Extract the character limit from the user message if specified."""
        match = re.search(r"in (\d+) characters?", user_message.lower())
        return int(match.group(1)) if match else None

    async def inlet(self, body: Dict, user: Optional[Dict] = None) -> Dict:
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
            # Use default max tokens for the category
            max_tokens = self.SETTINGS[category]["max_tokens"]

        # Get settings for the category
        settings = self.SETTINGS[category].copy()
        settings["max_tokens"] = max_tokens

        # Update body with essential settings
        body.update({
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
            "stop": settings["stop"]
        })

        # Append note to the last user message
        note = (
            f"\n\n[Note: Your request was categorized as '{category}'. "
            f"The response will be generated with these settings: "
            f"temperature={settings['temperature']}, max_tokens={settings['max_tokens']}, "
            f"stop={settings['stop']}]"
        )

        # Modify the last user message
        if messages:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i]["content"] += note
                    break
            else:
                messages.append({"role": "user", "content": user_message + note})
            body["messages"] = messages

        logger.info(f"Modified body: {body}")
        return body

if __name__ == "__main__":
    body = {
        "messages": [{"role": "user", "content": "Write a short Story in one hundred characters?"}]
    }
    pipeline = Pipeline()
    import asyncio
    result = asyncio.run(pipeline.inlet(body))
    print(result)
