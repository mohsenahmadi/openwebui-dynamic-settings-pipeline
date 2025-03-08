"""
Open WebUI Dynamic Parameter Pipeline
- Automatic content classification
- Category-specific LLM parameter injection
- Transparent response tagging
"""

from fastapi import APIRouter, Request, HTTPException
import re
import logging
from typing import Dict, Any

router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================
# CATEGORY CONFIGURATION
# =====================
CATEGORY_SETTINGS = {
    "creative_writing": {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
        "max_tokens": 1000
    },
    "technical_code": {
        "temperature": 0.3,
        "top_p": 0.5,
        "top_k": 20,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "max_tokens": 800
    },
    "business_docs": {
        "temperature": 0.4,
        "top_p": 0.7,
        "top_k": 40,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2,
        "max_tokens": 600
    },
    "general": {
        "temperature": 0.5,
        "top_p": 0.8,
        "top_k": 40,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.1,
        "max_tokens": 800
    }
}

# ===================
# CLASSIFICATION LOGIC
# ===================
def classify_request(prompt: str) -> str:
    """Determine content category using regex patterns"""
    prompt = prompt.lower()
    
    patterns = {
        "creative_writing": r"\b(story|plot|character|scene|chapter|novel)\b",
        "technical_code": r"\b(code|function|algorithm|debug|script|python|javascript)\b",
        "business_docs": r"\b(report|proposal|executive|strategy|board|meeting minutes)\b"
    }

    for category, pattern in patterns.items():
        if re.search(pattern, prompt):
            return category
            
    return "general"

# ======================
# CORE PROCESSING PIPELINE
# ======================
def llm_generate(prompt: str, parameters: Dict[str, Any]) -> str:
    """
    Mock LLM generation function
    In actual implementation, replace with your model invocation
    """
    # Replace this with actual model call:
    return f"Generated response for: {prompt}"

@router.post("/v1/dynamic-generate")
async def dynamic_generation(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "").strip()
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Empty prompt")
        
        # 1. Classify request
        category = classify_request(prompt)
        logger.info(f"Classified '{prompt[:30]}...' as: {category}")
        
        # 2. Get parameters
        params = CATEGORY_SETTINGS.get(category, CATEGORY_SETTINGS["general"])
        
        # 3. Generate response
        response = llm_generate(prompt, params)
        
        # 4. Add classification notice
        notice = f"[System: Using {category.replace('_', ' ')} settings - Temp={params['temperature']}, Top_k={params['top_k']}]"
        formatted_response = f"{notice}\n\n{response}"
        
        return {"response": formatted_response}
    
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail="Generation error")

# ======================
# INTEGRATION HELPERS
# ======================
def setup_openwebui_integration(app):
    """Integrate with Open WebUI's existing FastAPI app"""
    app.include_router(router)
    logger.info("Dynamic pipeline integrated with Open WebUI")
