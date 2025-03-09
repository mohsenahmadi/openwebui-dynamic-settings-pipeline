# categorize_input.py
# Pipeline to categorize user input and tag it with model/settings recommendations

CATEGORIES = {
    "Creative Writing": ("anthropic/claude-3.7-sonnet", {"temperature": 0.9, "top_p": 0.95, "top_k": 50}),
    "Technical Writing": ("gpt-4.5", {"temperature": 0.3, "top_p": 0.8, "top_k": 40}),
    "Business Writing": ("gpt-4.5", {"temperature": 0.5, "top_p": 0.85, "top_k": 40}),
    "Educational Content": ("gpt-4.5", {"temperature": 0.4, "top_p": 0.8, "top_k": 40}),
    "Social Media Posts": ("anthropic/claude-3.7-sonnet", {"temperature": 0.8, "top_p": 0.9, "top_k": 50}),
    "Translation": ("gpt-4.5", {"temperature": 0.2, "top_p": 0.7, "top_k": 30}),
    "Summarization": ("gpt-4.5", {"temperature": 0.3, "top_p": 0.8, "top_k": 40}),
    "Question Answering": ("gpt-4.5", {"temperature": 0.4, "top_p": 0.85, "top_k": 40}),
    "General": ("gpt-4.5", {"temperature": 0.6, "top_p": 0.9, "top_k": 50}),
}

def categorize_input(input_data: dict) -> dict:
    """
    Analyzes user input, categorizes it, and tags it with model/settings.
    Args:
        input_data (dict): The input payload containing the user's message.
    Returns:
        dict: Updated input_data with tags for category, model, and settings.
    """
    question = input_data.get("content", "").lower()
    
    # Keyword-based categorization (customize as needed)
    if "write a story" in question or "creative" in question:
        category = "Creative Writing"
    elif "technical" in question or "code" in question or "document" in question:
        category = "Technical Writing"
    elif "business" in question or "proposal" in question or "email" in question:
        category = "Business Writing"
    elif "educat" in question or "teach" in question or "lesson" in question:
        category = "Educational Content"
    elif "social media" in question or "post" in question or "tweet" in question:
        category = "Social Media Posts"
    elif "translate" in question or "language" in question:
        category = "Translation"
    elif "summarize" in question or "summary" in question:
        category = "Summarization"
    elif "what" in question or "how" in question or "why" in question:
        category = "Question Answering"
    else:
        category = "General"

    # Get model and settings
    model, settings = CATEGORIES[category]
    
    # Add tags to input_data
    input_data["tags"] = {
        "category": category,
        "model": model,
        "settings": settings,
        "original_input": question  # Preserve original input for the function
    }
    return input_data

# Register the pipeline (required for Open WebUI compatibility)
pipeline = categorize_input
