from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import re

class Pipeline:
    def __init__(self):
        # Set a human-readable name; id is inferred from filename
        self.name = "DynamicCategoryLLMAdjuster"

    async def on_startup(self):
        # This function is called when the server is started
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped
        print(f"on_shutdown:{__name__}")
        pass

    # Define category settings
    SETTINGS = {
        "Creative Writing": {
            "temperature": 1.0,
            "top_k": 75,
            "top_p": 0.9,
            "max_tokens": 750,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.6,
            "stop": []  # OpenAI-compatible 'stop' parameter
        },
        "Technical Writing": {
            "temperature": 0.2,
            "top_k": 15,
            "top_p": 0.5,
            "max_tokens": 300,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.1,
            "stop": ["\n\n", "```"]
        },
        "Business Documents": {
            "temperature": 0.6,
            "top_k": 40,
            "top_p": 0.7,
            "max_tokens": 200,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.5,
            "stop": ["Best regards", "Sincerely"]
        },
        "Conversational Responses": {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.8,
            "max_tokens": 150,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.4,
            "stop": ["\n"]
        },
        "Summarization": {
            "temperature": 0.4,
            "top_k": 25,
            "top_p": 0.6,
            "max_tokens": 100,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
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
            return "Conversational Responses"  # Default for general queries

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline logic to adjust LLM settings dynamically."""
        print(f"pipe:{__name__}")
        print(f"User message: {user_message}")
        print(f"Body: {body}")

        # Handle title generation if requested
        if body.get("title", False):
            print("Title Generation")
            return "Dynamic Category LLM Adjuster"

        # Detect category and get settings
        category = self.detect_category(user_message)
        settings = self.SETTINGS[category]

        # Update the body with LLM settings (OpenAI-compatible parameters)
        body.update({
            "temperature": settings["temperature"],
            "top_k": settings["top_k"],
            "top_p": settings["top_p"],
            "max_tokens": settings["max_tokens"],
            "frequency_penalty": settings["frequency_penalty"],
            "presence_penalty": settings["presence_penalty"],
            "stop": settings["stop"]
        })

        # Append note to the prompt
        note = (
            f"\n\n[Note: Your request was categorized as '{category}'. "
            f"The response will be generated with these settings: "
            f"temperature={settings['temperature']}, top_k={settings['top_k']}, "
            f"top_p={settings['top_p']}, max_tokens={settings['max_tokens']}, "
            f"frequency_penalty={settings['frequency_penalty']}, "
            f"presence_penalty={settings['presence_penalty']}, "
            f"stop={settings['stop']}]"
        )
        modified_prompt = user_message + note

        # Update the body with the modified prompt
        body["messages"] = messages  # Preserve existing messages
        body["messages"].append({"role": "user", "content": modified_prompt})

        # Return the modified prompt; Open WebUI will handle the generation
        print(f"Modified body: {body}")
        return modified_prompt
