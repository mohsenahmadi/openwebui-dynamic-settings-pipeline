from pydantic import BaseModel, Extra

# A simple wrapper model that includes a 'body' field
# and provides a model_dump() method that returns the wrapped dictionary.
class RequestWrapper(BaseModel):
    body: dict

    def model_dump(self, *args, **kwargs):
        return self.body

    class Config:
        extra = Extra.allow

class Pipeline:
    def __init__(self):
        # Pipeline name for identification.
        self.name = "AutoParamAdjust"
        # Define the pipeline type ("filter" means it modifies requests before they reach the LLM).
        self.type = "filter"
        # Valves: Apply this filter to all pipelines (all conversations) with high priority (0).
        self.valves = {"pipelines": ["*"], "priority": 0}

    async def inlet(self, body: dict, user: dict = None):
        """
        This inlet function intercepts the incoming request before it is sent to the LLM.
        It examines the latest user message, categorizes it, and adjusts generation parameters accordingly.
        """
        try:
            messages = body.get("messages", [])
            if messages:
                # Get the last user message (assuming the last message is the user's prompt)
                prompt = messages[-1].get("content", "").lower()

                # Determine the category based on simple keyword matching.
                category = "general"
                if any(kw in prompt for kw in ["story", "poem", "imagine"]):
                    category = "creative"
                elif any(kw in prompt for kw in ["explain", "what is", "how to", "definition"]):
                    category = "technical"
                elif any(kw in prompt for kw in ["code", "function", "algorithm"]):
                    category = "code"

                # Adjust generation parameters based on the determined category.
                if category == "creative":
                    body["temperature"] = 0.9    # High randomness for creative content.
                    body["top_p"] = 0.95         # Broader token selection.
                    body["max_tokens"] = 1024    # Allow longer responses.
                elif category == "technical":
                    body["temperature"] = 0.2    # Low randomness for factual accuracy.
                    body["top_p"] = 0.8
                    body["max_tokens"] = 512
                elif category == "code":
                    body["temperature"] = 0.1    # Very deterministic for code generation.
                    body["top_p"] = 0.5
                    body["max_tokens"] = 256
                else:
                    # Default settings for general queries.
                    body["temperature"] = 0.7
                    body["top_p"] = 0.9
                    body["max_tokens"] = 768
        except Exception as e:
            print("Error in AutoParamAdjust pipeline inlet:", e)
        # Return the modified body wrapped in our RequestWrapper model.
        return RequestWrapper(body=body)

    async def outlet(self, body: dict, user: dict = None):
        """
        The outlet function processes the response from the LLM.
        In this case, it simply wraps the output so that it supports model_dump().
        """
        return RequestWrapper(body=body)
