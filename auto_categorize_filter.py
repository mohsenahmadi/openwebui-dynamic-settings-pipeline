"""
AutoCategorizeFilter

This filter classifies user messages into categories
and adjusts generation parameters accordingly,
using only Python dict access (no model_dump calls).
"""

class AutoCategorizeFilter:
    def __init__(self):
        """
        Runs once when the filter is initialized.
        We do not use Pydantic or zero-shot classification here,
        so there's nothing to load.
        """
        pass

    def inlet(self, body: dict, __user__: dict = None) -> dict:
        """
        This method is called for each new user request
        before sending it to the model.

        'body' is a normal Python dictionary containing:
          - messages: The conversation list
          - temperature, top_p, max_tokens, etc. if present
        We classify the user's message (rule-based) and set parameters.
        """
        # 1. Extract the user's latest message text
        user_message = body["messages"][-1]["content"].lower()

        # 2. Classify the user's message into a category (rule-based example)
        if "summarize" in user_message or "summary" in user_message:
            category = "Summarization"
        elif "translate" in user_message or "translation" in user_message:
            category = "Translation"
        elif "write a story" in user_message or "poem" in user_message or "creative" in user_message:
            category = "Creative Writing"
        elif "explain" in user_message or "educational" in user_message:
            category = "Educational Content"
        elif "tweet" in user_message or "social media" in user_message or "post this" in user_message:
            category = "Social Media Posts"
        elif "business" in user_message or "professional email" in user_message:
            category = "Business Writing"
        elif user_message.endswith("?") or "how to" in user_message:
            category = "Question Answering"
        elif "technical" in user_message or "code" in user_message or "error" in user_message:
            category = "Technical Writing"
        else:
            category = "General"

        # 3. Adjust the generation parameters directly on body (no model_dump)
        if category == "Creative Writing":
            body["temperature"] = 0.9
            body["top_p"] = 0.95
            body["top_k"] = 0
            body["frequency_penalty"] = 0.0
            body["presence_penalty"] = 0.0
            body["max_tokens"] = 1024

        elif category == "Technical Writing" or category == "Question Answering":
            body["temperature"] = 0.2
            body["top_p"] = 0.8
            body["top_k"] = 40
            body["frequency_penalty"] = 0.2
            body["presence_penalty"] = 0.0
            body["max_tokens"] = 512

        elif category == "Business Writing":
            body["temperature"] = 0.4
            body["top_p"] = 0.9
            body["top_k"] = 40
            body["frequency_penalty"] = 0.1
            body["presence_penalty"] = 0.0
            body["max_tokens"] = 400

        elif category == "Educational Content":
            body["temperature"] = 0.5
            body["top_p"] = 0.9
            body["top_k"] = 0
            body["frequency_penalty"] = 0.0
            body["presence_penalty"] = 0.0
            body["max_tokens"] = 800

        elif category == "Social Media Posts":
            body["temperature"] = 0.7
            body["top_p"] = 0.95
            body["top_k"] = 0
            body["frequency_penalty"] = 0.0
            body["presence_penalty"] = 0.0
            body["max_tokens"] = 280

        elif category == "Translation":
            body["temperature"] = 0.1
            body["top_p"] = 1.0
            body["top_k"] = 0
            body["frequency_penalty"] = 0.0
            body["presence_penalty"] = 0.0
            body["max_tokens"] = 512

        elif category == "Summarization":
            body["temperature"] = 0.3
            body["top_p"] = 0.9
            body["top_k"] = 0
            body["frequency_penalty"] = 0.0
            body["presence_penalty"] = 0.3
            body["max_tokens"] = 200

        else:  # General
            body["temperature"] = 0.7
            body["top_p"] = 0.9
            body["top_k"] = 40
            body["frequency_penalty"] = 0.0
            body["presence_penalty"] = 0.0
            body["max_tokens"] = 512

        # Optional: Print debug info to console logs
        print(f"[AutoCategorizeFilter] Category = {category}")
        print(f"[AutoCategorizeFilter] Updated params: "
              f"temperature={body['temperature']}, top_p={body['top_p']}, top_k={body['top_k']}")

        # Return the updated dictionary
        return body

    def outlet(self, response: dict, __user__: dict = None) -> dict:
        """
        Called after the model responds, if you want to modify
        the final output. We leave it unchanged.
        """
        return response
