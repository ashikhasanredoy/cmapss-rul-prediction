import sys
import pandas as pd
from typing import List, Union
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.logger import logger
from src.exception import CustomException
from src.inference.predict import RULPredictor
from api.schemas import (
    EngineCycleData,
    PredictionResponse,
    BatchPredictionRequest,
    HealthStatusResponse
)

app = FastAPI(
    title="NASA Turbofan RUL Prediction API",
    description="Predictive Maintenance API for Turbofan Engine Remaining Useful Life",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    predictor = RULPredictor()
    logger.info("FastAPI: RULPredictor loaded successfully")
except Exception as e:
    predictor = None
    logger.error(f"FastAPI: Failed to initialize RULPredictor: {e}")

def dump_model(model_obj):
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump()
    return model_obj.dict()

@app.get("/", response_model=HealthStatusResponse)
def root():
    return HealthStatusResponse(
        status="active" if predictor is not None else "model_not_loaded",
        model_version="1.0.0 (Stacking Ensemble)",
        dataset="NASA C-MAPSS FD001"
    )

@app.get("/health", response_model=HealthStatusResponse)
def health_check():
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready or artifacts are missing."
        )
    return HealthStatusResponse(
        status="healthy",
        model_version="1.0.0",
        dataset="NASA C-MAPSS FD001"
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_rul(data: Union[EngineCycleData, List[EngineCycleData]]):
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor artifact is not initialized."
        )
    try:
        if isinstance(data, list):
            records = [dump_model(item) for item in data]
        else:
            records = [dump_model(data)]

        df = pd.DataFrame(records)
        result = predictor.predict_engine(df)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/predict/batch", response_model=PredictionResponse)
def predict_batch_rul(request: BatchPredictionRequest):
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor artifact is not initialized."
        )
    try:
        records = [dump_model(item) for item in request.records]
        df = pd.DataFrame(records)
        result = predictor.predict_engine(df)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
