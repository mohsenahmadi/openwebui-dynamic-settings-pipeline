import json

def load_tag_params():
    try:
        with open("tag_params.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

tag_params = load_tag_params()

def post_user_message(request, session):
    # Assuming request and session are dictionaries
    messages = request.get("messages", [])
    if messages:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content", "").startswith("Tag: "):
                tag = msg["content"].split("Tag: ", 1)[1].strip()
                session["tag"] = tag
                break
    return request, session

def pre_llm(request, session):
    tag = session.get("tag")
    if tag and tag in tag_params:
        params = tag_params[tag]
        # Set standard parameters directly in the request dictionary
        standard_params = ["temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"]
        for param in standard_params:
            if param in params:
                request[param] = params[param]
        # Handle custom parameters like reasoning_effort
        if "reasoning_effort" in params:
            reasoning = params["reasoning_effort"]
            instruction = ""
            if reasoning == "high":
                instruction = "\nPlease provide a detailed step-by-step reasoning."
            elif reasoning == "medium":
                instruction = "\nPlease provide some reasoning."
            # Modify the prompt or messages
            if "prompt" in request and isinstance(request["prompt"], str):
                request["prompt"] += instruction
            elif "messages" in request and isinstance(request["messages"], list):
                if request["messages"] and isinstance(request["messages"][-1], dict) and request["messages"][-1].get("role") == "user":
                    request["messages"][-1]["content"] += instruction
                else:
                    request["messages"].append({"role": "user", "content": instruction})
    return request
