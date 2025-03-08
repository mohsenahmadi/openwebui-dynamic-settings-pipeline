import json

def load_tag_params():
    try:
        with open("tag_params.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

tag_params = load_tag_params()

def post_user_message(request, session):
    messages = request.get("messages")
    if messages:
        for msg in messages:
            if msg["role"] == "user" and msg["content"].startswith("Tag: "):
                tag = msg["content"].split("Tag: ", 1)[1].strip()
                session["tag"] = tag
                break
    return request, session

def pre_llm(request, session):
    tag = session.get("tag")
    if tag and tag in tag_params:
        params = tag_params[tag]
        # Set standard parameters
        standard_params = ["temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"]
        for param in standard_params:
            if param in params:
                request[param] = params[param]
        # Handle custom parameters
        if "reasoning_effort" in params:
            reasoning = params["reasoning_effort"]
            instruction = ""
            if reasoning == "high":
                instruction = "\nPlease provide a detailed step-by-step reasoning."
            elif reasoning == "medium":
                instruction = "\nPlease provide some reasoning."
            # Modify the prompt accordingly
            if isinstance(request.get("prompt"), str):
                request["prompt"] += instruction
            elif isinstance(request.get("messages"), list):
                # Assuming it's a list of messages, and the last message is from the user
                if request["messages"] and request["messages"][-1]["role"] == "user":
                    request["messages"][-1]["content"] += instruction
                else:
                    # Add a new user message with the instruction
                    request["messages"].append({"role": "user", "content": instruction})
    return request
