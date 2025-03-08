"""
title: Dynamic Settings Manifold Pipeline
author: D!Mo
date: 2025-03-08
version: 1.1
license: MIT
description: A manifold pipeline that dynamically adjusts OpenAI API settings and appends the applied settings map to the response.
requirements: None
"""

from openwebui.pipelines import pipe

# Define categories and their keywords
categories = {
    'Creative Writing': ['story', 'novel', 'poem', 'character', 'plot', 'fiction', 'imagination'],
    'Technical Writing': ['code', 'script', 'documentation', 'API', 'technical', 'programming'],
    'Business Writing': ['report', 'proposal', 'email', 'business', 'marketing', 'sales'],
    'Educational Content': ['lecture', 'tutorial', 'explanation', 'teach', 'learn', 'education'],
    'Social Media Posts': ['tweet', 'post', 'caption', 'social media', 'viral', 'engagement'],
    'Translation': ['translate', 'from', 'to', 'language', 'translation'],
    'Summarization': ['summarize', 'summary', 'condense', 'key points'],
    'Question Answering': ['what', 'who', 'when', 'where', 'why', 'how', 'question', 'answer']
}

# Define settings for each category
settings = {
    'Creative Writing': {
        'temperature': 0.9,
        'top_p': 0.9,
        'max_tokens': 500,
        'frequency_penalty': 0.2,
        'presence_penalty': 0.2,
        'reasoning_effort': 'medium'
    },
    'Technical Writing': {
        'temperature': 0.2,
        'top_p': 0.5,
        'max_tokens': 200,
        'frequency_penalty': 0.5,
        'presence_penalty': 0.5,
        'reasoning_effort': 'medium'
    },
    'Business Writing': {
        'temperature': 0.6,
        'top_p': 0.75,
        'max_tokens': 300,
        'frequency_penalty': 0.3,
        'presence_penalty': 0.3,
        'reasoning_effort': 'high'
    },
    'Educational Content': {
        'temperature': 0.5,
        'top_p': 0.7,
        'max_tokens': 400,
        'frequency_penalty': 0.4,
        'presence_penalty': 0.4,
        'reasoning_effort': 'high'
    },
    'Social Media Posts': {
        'temperature': 0.75,
        'top_p': 0.85,
        'max_tokens': 50,
        'frequency_penalty': 0.1,
        'presence_penalty': 0.1,
        'reasoning_effort': 'low'
    },
    'Translation': {
        'temperature': 0.1,
        'top_p': 0.5,
        'max_tokens': 100,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'reasoning_effort': 'low'
    },
    'Summarization': {
        'temperature': 0.6,
        'top_p': 0.75,
        'max_tokens': 200,
        'frequency_penalty': 0.3,
        'presence_penalty': 0.3,
        'reasoning_effort': 'medium'
    },
    'Question Answering': {
        'temperature': 0.2,
        'top_p': 0.5,
        'max_tokens': 100,
        'frequency_penalty': 0.2,
        'presence_penalty': 0.2,
        'reasoning_effort': 'medium'
    },
    'General': {
        'temperature': 0.5,
        'top_p': 0.7,
        'max_tokens': 250,
        'frequency_penalty': 0.3,
        'presence_penalty': 0.3,
        'reasoning_effort': 'medium'
    }
}

def classify_user_input(user_input):
    user_input_lower = user_input.lower()
    
    # Check for "write a" or "create a"
    if user_input_lower.startswith('write a ') or user_input_lower.startswith('create a '):
        content_type = user_input_lower[len('write a '):].strip() if user_input_lower.startswith('write a ') else user_input_lower[len('create a '):].strip()
        for category, keywords in categories.items():
            if any(keyword in content_type for keyword in keywords):
                return category
        return 'Creative Writing'
    
    # Check for translation
    if 'translate' in user_input_lower and 'from' in user_input_lower and 'to' in user_input_lower:
        return 'Translation'
    
    # Check for question
    if user_input_lower.endswith('?'):
        return 'Question Answering'
    
    # Check for summarization
    if 'summarize' in user_input_lower or 'summary' in user_input_lower:
        return 'Summarization'
    
    # Check for social media posts
    if any(keyword in user_input_lower for keyword in categories['Social Media Posts']):
        return 'Social Media Posts'
    
    # Check for other categories based on keywords
    for category, keywords in categories.items():
        if any(keyword in user_input_lower for keyword in keywords):
            return category
    
    # Default category
    return 'General'

# Pipe to classify and set parameters before LLM call
@pipe(before=True)
def dynamic_settings_pipe(request):
    user_input = request['messages'][-1]['content']
    category = classify_user_input(user_input)
    params = settings[category]
    request['temperature'] = params['temperature']
    request['top_p'] = params['top_p']
    request['max_tokens'] = params['max_tokens']
    request['frequency_penalty'] = params['frequency_penalty']
    request['presence_penalty'] = params['presence_penalty']
    request['reasoning_effort'] = params['reasoning_effort']
    request['category'] = category
    return request

# Pipe to annotate the response after LLM call
@pipe(after=True)
def annotate_response_pipe(request, response):
    category = request.get('category', 'Unknown')
    params = settings.get(category, {
        'temperature': None,
        'top_p': None,
        'max_tokens': None,
        'frequency_penalty': None,
        'presence_penalty': None,
        'reasoning_effort': None
    })
    note = f"Note: This response was generated for the category '{category}' with temperature {params['temperature']}, top_p {params['top_p']}, max_tokens {params['max_tokens']}, frequency_penalty {params['frequency_penalty']}, presence_penalty {params['presence_penalty']}, and reasoning_effort '{params['reasoning_effort']}'."
    response['content'] += "\n\n" + note
    return response
