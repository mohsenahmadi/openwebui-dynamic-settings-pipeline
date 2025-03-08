import re
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from schemas import OpenAIChatMessage


class Pipeline:
    """
    Example pipeline that extracts a [CATEGORY] ... [/CATEGORY] tag
    from the model output and stores it in the 'metadata' field.
    """

    class Valves(BaseModel):
        # You can define custom valves (settings) here if desired.
        pass

    def __init__(self):
        """
        Optionally set an ID or name for the pipeline. 
        Best practice: let ID be inferred from filename.
        """
        self.name = "Pipeline Example with Category Extraction"
        # self.id = "pipeline_example_category"  # optionally specify a unique ID

    async def on_startup(self):
        """
        Called when the server is started.
        """
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        """
        Called when the server is stopped.
        """
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        """
        Called when valves (settings) are updated.
        """
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        """
        Called before the OpenAI API request is made.
        You can modify the form data before it is sent to the API.
        """
        print(f"inlet: {__name__}")
        # Debug prints for demonstration
        print("inlet body:", body)
        print("inlet user:", user)

        # You can manipulate 'body' here if needed.
        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        """
        Called after the OpenAI API response is completed.
        You can inspect or modify the response here before returning to the client.
        """
        print(f"outlet: {__name__}")
        # Debug prints for demonstration
        print("outlet body (before):", body)
        print("outlet user:", user)

        # Here, we assume body follows the standard OpenAI response format:
        # {
        #   "choices": [
        #       {
        #           "message": {
        #               "content": "<the final model output>"
        #           }
        #       }
        #   ],
        #   ...
        # }
        #
        # You can adjust as necessary if your format is different.

        # Make sure there is at least one choice with content
        if "choices" in body and len(body["choices"]) > 0:
            content = body["choices"][0].get("message", {}).get("content", "")

            # Look for [CATEGORY] ... [/CATEGORY]
            match = re.search(r"\[CATEGORY\](.*?)\[/CATEGORY\]", content, re.DOTALL)
            if match:
                extracted_category = match.group(1).strip()

                # Store in body["metadata"] (create if it doesn't exist)
                if "metadata" not in body:
                    body["metadata"] = {}
                body["metadata"]["category"] = extracted_category

                # Optional: Remove the tag from the final content
                cleaned_content = re.sub(r"\[CATEGORY\].*?\[/CATEGORY\]", "", content, flags=re.DOTALL).strip()
                body["choices"][0]["message"]["content"] = cleaned_content

        print("outlet body (after):", body)
        return body

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        This is where you can add your custom pipeline logic (e.g., RAG or other).
        For demonstration, it returns a string indicating it's from this pipeline.
        """
        print(f"pipe: {__name__}")

        # Example check: if title generation is requested
        if body.get("title", False):
            print("Title Generation Request")

        # Debug prints
        print("messages:", messages)
        print("user_message:", user_message)
        print("body:", body)

        # Return any string or generator you wish. 
        # For demonstration, we simply return a placeholder response.
        return f"{__name__} response to: {user_message}"
