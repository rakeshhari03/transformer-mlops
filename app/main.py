import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model_service import ModelService

app = FastAPI(
    title="Transformer Sentiment API",
    version="1.0.0"
)

model_service = None

class PredictionRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=10000
    )

@app.on_event("startup")
def load_model():

    global model_service

    model_path = os.getenv(
        "MODEL_PATH",
        "./model/best_model"
    )

    model_service = ModelService(
        model_path
    )
    

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model_service is not None
    }

@app.post("/predict")
def predict(request: PredictionRequest):

    if model_service is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not ready"
        )

    return model_service.predict(
        request.text
    )