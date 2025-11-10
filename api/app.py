"""
FastAPI Application
Endpoints for inference and feature retrieval
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import os

from .inference import InferenceService
from storage.feature_store import FeatureStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for fraud detection using Graph Neural Networks",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
inference_service: Optional[InferenceService] = None
feature_store: Optional[FeatureStore] = None


# Pydantic models
class PredictionRequest(BaseModel):
    node_ids: List[str]
    graph_path: Optional[str] = None


class PredictionResponse(BaseModel):
    node_ids: List[str]
    predictions: List[int]
    probabilities: List[List[float]]
    fraud_scores: List[float]


class FeatureRequest(BaseModel):
    node_ids: List[str]
    feature_types: Optional[List[str]] = None


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global inference_service, feature_store
    
    logger.info("Starting up API services...")
    
    # Initialize feature store
    backend_type = os.getenv('FEATURE_STORE_BACKEND', 'csv')
    feature_store = FeatureStore(backend_type=backend_type)
    logger.info(f"Initialized feature store with backend: {backend_type}")
    
    # Initialize inference service if model exists
    model_path = os.getenv('MODEL_PATH', './model/model.pth')
    metadata_path = os.getenv('METADATA_PATH', './model/metadata.pkl')
    device = os.getenv('DEVICE', 'cpu')
    
    if os.path.exists(model_path) and os.path.exists(metadata_path):
        try:
            inference_service = InferenceService(
                model_path=model_path,
                metadata_path=metadata_path,
                feature_store=feature_store,
                device=device
            )
            logger.info("Initialized inference service")
        except Exception as e:
            logger.error(f"Failed to initialize inference service: {e}")
    else:
        logger.warning(f"Model files not found: {model_path}, {metadata_path}")
        logger.warning("Inference service will not be available")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "inference_service": inference_service is not None,
        "feature_store": feature_store is not None
    }


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict fraud for given node IDs
    
    Args:
        request: Prediction request with node IDs
        
    Returns:
        Prediction results
    """
    if inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not available")
    
    try:
        results = inference_service.predict(request.node_ids)
        return PredictionResponse(**results)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Feature retrieval endpoint
@app.post("/features")
async def get_features(request: FeatureRequest):
    """
    Get features for given node IDs
    
    Args:
        request: Feature request with node IDs
        
    Returns:
        Features as dictionary
    """
    if feature_store is None:
        raise HTTPException(status_code=503, detail="Feature store not available")
    
    try:
        features_df = inference_service.get_features(
            request.node_ids,
            request.feature_types
        ) if inference_service else feature_store.load_features('graph_features')
        
        # Convert to dictionary
        features_dict = features_df.to_dict(orient='index')
        return {
            "node_ids": request.node_ids,
            "features": features_dict
        }
    except Exception as e:
        logger.error(f"Feature retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Feature retrieval failed: {str(e)}")


# Get feature endpoint (GET version)
@app.get("/features")
async def get_features_get(
    node_ids: List[str] = Query(..., description="List of node IDs"),
    feature_types: Optional[List[str]] = Query(None, description="Feature types to retrieve")
):
    """
    Get features for given node IDs (GET version)
    
    Args:
        node_ids: List of node IDs
        feature_types: Optional feature types
        
    Returns:
        Features as dictionary
    """
    request = FeatureRequest(node_ids=node_ids, feature_types=feature_types)
    return await get_features(request)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Predict fraud (POST)",
            "/features": "Get features (POST/GET)"
        }
    }










