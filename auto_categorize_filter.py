"""
AutoCategorizeFilter

A single-file filter for Open WebUI that:
1. Inspects the user's latest message.
2. Classifies it (using simple keyword rules).
3. Adjusts generation parameters (temperature, top_p, etc.).
4. Returns the updated request body.

No calls to model_dump() are present.
"""

class AutoCategorizeFilter:
    def __init__(self):
        """
        Called once when this filter is loaded.
        We do no special initialization here,
        and we do NOT use pydantic or model_dump.
        """
        pass

    def inlet(self, body: dict, __user__: dict = None) -> dict:
        """
        Called for each new user request before it's sent to the model.

        `body` is a normal Python dictionary containing request data:
          - messages: The conversation list (list of dicts)
          - temperature, top_p, max_tokens, etc.

        Returns the same `body` dict, possibly modified.
        """
        # 1. Extract the user's latest message content (lowercased for simple checking).
        user_message = body["messages"][-1]["content"].lower()

        # 2. Assign a category based on simple keywords.
        if "summarize" in user_message or "summary" in user_message:
            category = "Summarization"
        elif "translate" in user_message or "translation" in user_message:
            category = "Translation"
        elif "write a story" in user_message or "poem" in user_message or "creative" in user_message:
            category = "Creative Writing"
        elif "explain" in user_message or "educational" in user_message:
            category = "Educational Content"
        elif "tweet" in user_message or "social media" in user_message or "post" in user_message:
            category = "Social Media Posts"
        elif "business" in user_message or "professional email" in user_message:
            category = "Business Writing"
        elif user_message.endswith("?") or "how to" in user_message:
            category = "Question Answering"
        elif "technical" in user_message or "code" in user_message or "error" in user_message:
            category = "Technical Writing"
        else:
            category = "General"

        # 3. Adjust Open WebUI generation parameters based on the category.
        #    None of these lines call model_dump(). We just modify the dict.
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

        # 4. Debug print statements (optional) to see what happened
        print(f"[AutoCategorizeFilter] Category: {category}")
        print(f"[AutoCategorizeFilter] Updated params: temperature={body.get('temperature')}, "
              f"top_p={body.get('top_p')}, top_k={body.get('top_k')}")

        # 5. Return the updated dict so Open WebUI can proceed with generation.
        return body

    def outlet(self, response: dict, __user__: dict = None) -> dict:
        """
        Called after the model responds, letting us modify the
        final response if we want. We'll leave it as-is.
        """
        return response
