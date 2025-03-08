from pydantic import BaseModel

# A simple wrapper that provides the model_dump() method required by Open WebUI.
class DummyRequest(BaseModel):
    __root__: dict

    def model_dump(self, *args, **kwargs):
        return self.__root__

class Pipeline:
    def __init__(self):
        # Pipeline name for identification
        self.name = "AutoParamAdjust"
        # Define the type of this pipeline; "filter" means it modifies requests before they reach the LLM.
        self.type = "filter"
        # Valves: Apply this filter to all pipelines (all conversations) with high priority (0)
        self.valves = {"pipelines": ["*"], "priority": 0}

    async def inlet(self, body: dict, user: dict = None):
        """
        Intercepts the incoming request body before it is sent to the LLM.
        It examines the latest user message, categorizes it, and dynamically adjusts generation parameters.
        """
        try:
            messages = body.get("messages", [])
            if messages:
                # Retrieve the last message (assuming it's from the user)
                prompt = messages[-1].get("content", "").lower()

                # Determine the category using basic keyword matching.
                category = "general"
                if any(kw in prompt for kw in ["story", "poem", "imagine"]):
                    category = "creative"
                elif any(kw in prompt for kw in ["explain", "what is", "how to", "definition"]):
                    category = "technical"
                elif any(kw in prompt for kw in ["code", "function", "algorithm"]):
                    category = "code"

                # Adjust generation parameters based on the determined category
                if category == "creative":
                    body["temperature"] = 0.9    # High randomness for creative content
                    body["top_p"] = 0.95         # More diversity in token selection
                    body["max_tokens"] = 1024    # Allow longer responses
                elif category == "technical":
                    body["temperature"] = 0.2    # Low randomness for factual accuracy
                    body["top_p"] = 0.8
                    body["max_tokens"] = 512
                elif category == "code":
                    body["temperature"] = 0.1    # Very deterministic output for code generation
                    body["top_p"] = 0.5
                    body["max_tokens"] = 256
                else:
                    # Default settings for general queries
                    body["temperature"] = 0.7
                    body["top_p"] = 0.9
                    body["max_tokens"] = 768
        except Exception as e:
            # Log any error to assist with debugging
            print("Error in AutoParamAdjust pipeline inlet:", e)
        # Return the modified body wrapped in DummyRequest so it supports model_dump()
        return DummyRequest(__root__=body)

    async def outlet(self, body: dict, user: dict = None):
        """
        Processes the response from the LLM. In this case, it simply wraps the response
        so that it supports the model_dump() method.
        """
        return DummyRequest(__root__=body)
