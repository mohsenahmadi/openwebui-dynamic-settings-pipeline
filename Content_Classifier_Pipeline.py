"""
title: Content Classifier Pipeline
author: Claude
date: 2025-03-10
version: 1.0
license: MIT
description: A pipeline for classifying user messages into different content categories and adding appropriate tags to chat threads.
requirements: requests, huggingface_hub, openai
"""

from typing import List, Optional, Dict, Any
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests
import os
import json
from huggingface_hub import InferenceClient


class Pipeline:
    class Config(BaseModel):
        # API Keys for services
        huggingface_api_key: str = ""
        openai_api_key: str = ""
        webui_api_key: str = ""
        # Model selection
        huggingface_model: str = "facebook/bart-large-mnli"
        # Enable/disable fallback
        use_openai_fallback: bool = True
        
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        # e.g. ["llama3:latest", "gpt-3.5-turbo"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0
        
        # Configuration settings
        config: Optional["Pipeline.Config"] = None

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "content_classifier_pipeline"
        self.name = "Content Classifier"

        # Default configuration
        default_config = self.Config(
            huggingface_api_key="",
            openai_api_key="",
            webui_api_key="",
            huggingface_model="facebook/bart-large-mnli",
            use_openai_fallback=True
        )
        
        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "config": default_config
            }
        )
        
        # Base URL for Open WebUI API
        self.webui_api_url = "https://ai.dimo.page/api"
        
        # Content categories
        self.categories = [
            "Flirting & Seduction Writing",
            "Creative Writing",
            "Technical Writing",
            "Business Writing",
            "Development",
            "Educational Content",
            "Social Media Posts",
            "Translation",
            "Summarization",
            "Question Answering",
            "General"
        ]
        
        # Initialize HuggingFace client
        self.hf_client = None

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        
        try:
            if self.valves.config and self.valves.config.huggingface_api_key:
                self.hf_client = InferenceClient(token=self.valves.config.huggingface_api_key)
                print(f"HuggingFace client initialized successfully")
        except Exception as e:
            print(f"Error initializing HuggingFace client: {e}")
            self.hf_client = None

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        
        # Update client with new config
        try:
            if self.valves.config and self.valves.config.huggingface_api_key:
                self.hf_client = InferenceClient(token=self.valves.config.huggingface_api_key)
                print(f"HuggingFace client re-initialized with updated token")
        except Exception as e:
            print(f"Error updating HuggingFace client: {e}")
            self.hf_client = None
    
    async def classify_with_huggingface(self, text: str) -> str:
        """Classify text using HuggingFace model"""
        try:
            if not self.hf_client:
                raise Exception("HuggingFace client not initialized")
            
            # Get model from config
            model = self.valves.config.huggingface_model if self.valves.config else "facebook/bart-large-mnli"
                
            # For zero-shot classification
            response = self.hf_client.zero_shot_classification(
                text,
                candidate_labels=self.categories,
                model=model  # Model from config
            )
            
            # Get the highest probability category
            if response and 'labels' in response and 'scores' in response:
                # Make sure the returned category matches exactly one of our categories
                top_category_index = 0
                top_category = response['labels'][top_category_index]
                
                # Verify the category is in our exact list, if not try the next one
                while top_category not in self.categories and top_category_index < len(response['labels']) - 1:
                    top_category_index += 1
                    top_category = response['labels'][top_category_index]
                
                # If we found a matching category, return it
                if top_category in self.categories:
                    return top_category
                
                # Otherwise return General
                return "General"
            
            # Default category if classification fails
            return "General"
        except Exception as e:
            print(f"Error classifying with HuggingFace: {e}")
            return None
    
    async def classify_with_openai(self, text: str) -> str:
        """Fallback classification using OpenAI"""
        try:
            if not self.valves.config or not self.valves.config.openai_api_key:
                print("OpenAI API key not configured")
                return "General"
                
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.valves.config.openai_api_key}"
            }
            
            # Create a list of exact category names to enforce exact matching
            categories_str = '"' + '", "'.join(self.categories) + '"'
            
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": f"Classify the following text into EXACTLY ONE of these categories - no variations allowed: {categories_str}. Respond with ONLY the exact category name, no explanation or additions."},
                    {"role": "user", "content": text}
                ],
                "temperature": 0.3,
                "max_tokens": 30
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                category = result["choices"][0]["message"]["content"].strip()
                
                # Remove any quotes, periods or other punctuation that might have been added
                category = category.strip('"\'.,;:()[]{}')
                
                # Ensure the returned category is valid - exact match only
                if category in self.categories:
                    return category
                
                # If no exact match, return General
                return "General"
            else:
                print(f"OpenAI API error: {response.status_code}, {response.text}")
                return "General"
        except Exception as e:
            print(f"Error classifying with OpenAI: {e}")
            return "General"
    
    async def add_tag_to_thread(self, thread_id: str, tag: str) -> bool:
        """Add classification tag to the chat thread"""
        try:
            # Assuming there's an API endpoint to update thread tags
            # This would need to be adjusted based on Open WebUI's actual API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.webui_api_key}"
            }
            
            payload = {
                "thread_id": thread_id,
                "tag": tag
            }
            
            # This is a placeholder URL - you'll need to use the actual endpoint
            response = requests.post(
                f"{self.webui_api_url}/threads/tag",
                headers=headers,
                json=payload
            )
            
            if response.status_code in [200, 201]:
                print(f"Successfully added tag '{tag}' to thread {thread_id}")
                return True
            else:
                print(f"Failed to add tag: {response.status_code}, {response.text}")
                return False
        except Exception as e:
            print(f"Error adding tag to thread: {e}")
            return False

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Process incoming messages and classify content"""
        print(f"inlet:{__name__}")
        
        try:
            # Extract user message and thread ID
            user_message = body["messages"][-1]["content"]
            
            # Get thread ID if available
            thread_id = body.get("thread_id", None)
            if not thread_id and "thread" in body:
                thread_id = body["thread"].get("id", None)
            
            # Classify the message
            category = await self.classify_with_huggingface(user_message)
            
            # Use OpenAI as fallback if HuggingFace fails and fallback is enabled
            if not category and self.valves.config and self.valves.config.use_openai_fallback:
                category = await self.classify_with_openai(user_message)
            
            # If we still don't have a category, use default
            if not category:
                category = "General"
            
            # Verify the category is one of our exact predefined categories
            if category not in self.categories:
                print(f"Warning: Classification returned '{category}' which isn't in our categories list")
                category = "General"
                
            print(f"Message classified as: {category}")
            
            # Add tag to thread if thread_id is available
            if thread_id:
                success = await self.add_tag_to_thread(thread_id, category)
                print(f"Tag addition {'successful' if success else 'failed'}")
            
            # Add classification info to system message for the model's awareness
            # Only add if we have messages and the first one is a system message
            if body.get("messages") and len(body["messages"]) > 0 and body["messages"][0].get("role") == "system":
                current_system = body["messages"][0]["content"]
                updated_system = f"{current_system}\n\nThe user's request falls under the category: {category}."
                body["messages"][0]["content"] = updated_system
            
            # Return the modified request body
            return body
            
        except Exception as e:
            print(f"Error in content classifier pipeline: {e}")
            # If there's an error, return the original body unchanged
            return body
