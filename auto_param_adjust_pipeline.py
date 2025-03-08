# auto_param_adjust_pipeline.py

class ModelDumpDict(dict):
    """
    A custom dictionary subclass that implements the model_dump method.
    This ensures the returned object has the model_dump() method expected by Open WebUI.
    """
    def model_dump(self, *args, **kwargs):
        return self

class Pipeline:
    def __init__(self):
        # Pipeline name for identification
        self.name = "AutoParamAdjust"
        # Define the pipeline type ("filter" means it modifies requests before they reach the LLM)
        self.type = "filter"
        # Valves: Apply this filter to all pipelines (all conversations) with high priority (0)
        self.valves = {"pipelines": ["*"], "priority": 0}

    async def inlet(self, body: dict, user: dict = None):
        """
        Intercepts the incoming request body before it is sent to the LLM.
        It examines the latest user message, categorizes it, and adjusts generation parameters accordingly.
        """
        try:
            messages = body.get("messages", [])
            if messages:
                # Retrieve the last message (assuming it's from the user)
                prompt = messages[-1].get("content", "").lower()

                # Determine the category using simple keyword matching.
                category = "general"
                if any(kw in prompt for kw in ["story", "poem", "imagine"]):
                    category = "creative"
                elif any(kw in prompt for kw in ["explain", "what is", "how to", "definition"]):
                    category = "technical"
                elif any(kw in prompt for kw in ["code", "function", "algorithm"]):
                    category = "code"

                # Adjust generation parameters based on the determined category
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
        # Return the modified body wrapped in ModelDumpDict so it supports model_dump()
        return ModelDumpDict(body)

    async def outlet(self, body: dict, user: dict = None):
        """
        Processes the response from the LLM.
        In this case, it simply wraps the output so that it supports model_dump().
        """
        return ModelDumpDict(body)
