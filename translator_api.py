from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from translator import ONNXTranslator
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import os
from contextlib import asynccontextmanager

load_dotenv()
translator = None
MODEL_DIR = Path(os.getenv('MODEL_DIR'))

@asynccontextmanager
async def lifespan(app:FastAPI):
    global translator
    print("Loading ONNX translator...")
    translator = ONNXTranslator(MODEL_DIR)
    print("Model loaded successfully!")
    yield

app = FastAPI(title = 'English-Hindi Translation API',lifespan=lifespan)

class TranslationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 100

class TranslationResponse(BaseModel):
    source: str
    translation: str
    inference_time: float

@app.get('/')
async def root():
    return{
        "message": "English-Hindi Translation API",
        "status": "running",
        "endpoints": {
            "translate": "/translate",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "loaded"}

@app.post('/translate',response_model=TranslationResponse)
async def translate(request:TranslationRequest):
    if translator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        start_time = time.time()
        translation = translator.translate(request.text, max_len=request.max_length)
        inference_time = time.time() - start_time
        
        return TranslationResponse(
            source=request.text,
            translation=translation,
            inference_time=inference_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)