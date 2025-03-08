from pipelines import Pipe
from typing import Union, Generator, Iterator

def classify_task(prompt):
    prompt = prompt.lower()
    if any(keyword in prompt for keyword in ["story", "imagine", "create", "poem"]):
        return "creative"
    elif any(keyword in prompt for keyword in ["what is", "explain", "define", "history"]):
        return "factual"
    elif any(keyword in prompt for keyword in ["code", "program", "function", "debug"]):
        return "technical"
    return "default"

SETTINGS_MAP = {
    "creative": {
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 0.95,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
        "max_tokens": 1000
    },
    "factual": {
        "temperature": 0.2,
        "top_k": 10,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 100
    },
    "technical": {
        "temperature": 0.3,
        "top_k": 20,
        "top_p": 0.85,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 300
    },
    "default": {
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.2,
        "max_tokens": 400
    }
}

class DynamicSettingsPipe(Pipe):
    def __init__(self):
        super().__init__()
        self.name = "DynamicSettings"

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        prompt = body.get("messages", [{}])[-1].get("content", "")
        task = classify_task(prompt)
        settings = SETTINGS_MAP.get(task, SETTINGS_MAP["default"])
        body["temperature"] = settings["temperature"]
        body["top_k"] = settings["top_k"]
        body["top_p"] = settings["top_p"]
        body["frequency_penalty"] = settings["frequency_penalty"]
        body["presence_penalty"] = settings["presence_penalty"]
        body["max_tokens"] = settings["max_tokens"]
        return body
