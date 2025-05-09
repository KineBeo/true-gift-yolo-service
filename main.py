from fastapi import FastAPI, File, UploadFile, Body, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any, Union
import uvicorn
import numpy as np
from PIL import Image
import io
import os
import sys
import logging
from ultralytics import YOLO
import time
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import tempfile
import shutil
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("yolo-service")

# Define Pydantic models for API documentation
class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    processing_time: float = Field(..., description="Processing time in seconds")

class BatchPredictionRequest(BaseModel):
    urls: List[str] = Field(
        ..., 
        description="List of image URLs to classify",
        example=[
            "https://example.com/pho.jpg",
            "https://example.com/banh-mi.jpg"
        ]
    )
    timeout_per_image: Optional[float] = Field(15.0, description="Timeout in seconds for each image")

class BatchPredictionResponse(BaseModel):
    results: Dict[str, Any] = Field(..., description="Prediction results for each URL")
    processing_time: float = Field(..., description="Total processing time in seconds")
    success_count: int = Field(..., description="Number of successfully processed images")
    error_count: int = Field(..., description="Number of failed images")

# Create FastAPI app
app = FastAPI(
    title="YOLO Vietnamese Food Classification API",
    description="API for classifying Vietnamese food images using YOLOv8",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory if it doesn't exist
TEMP_DIR = os.path.join(os.getcwd(), "temp")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
    logger.info(f"Created temporary directory: {TEMP_DIR}")

# Model path
MODEL_PATH = os.path.join('data', 'yolov8-vn-food-classification.pt')

# Check if model exists
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file not found at: {MODEL_PATH}")
    # List files in data directory for debugging
    data_dir = 'data'
    if os.path.exists(data_dir):
        logger.info(f"Files in {data_dir} directory:")
        for file in os.listdir(data_dir):
            logger.info(f"  - {file} ({os.path.getsize(os.path.join(data_dir, file))} bytes)")
        
        # Try to find the model file with any name
        for file in os.listdir(data_dir):
            if file.endswith('.pt'):
                MODEL_PATH = os.path.join(data_dir, file)
                logger.info(f"Found model file: {MODEL_PATH}")
                break
    else:
        logger.error(f"Data directory does not exist: {data_dir}")

# Initialize model
try:
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

@app.get("/", summary="Root endpoint", description="Check if the API is running")
async def root():
    return {"message": "YOLO Vietnamese Food Classification API", "status": "running"}

@app.get("/status", summary="Check API status", description="Get information about the API and model status")
async def status():
    model_loaded = model is not None
    return {
        "status": "ok" if model_loaded else "error",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH) if MODEL_PATH else False
    }

@app.post(
    "/predict", 
    summary="Predict food class from uploaded image", 
    description="Upload an image file to classify Vietnamese food",
    response_model=PredictionResponse,
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {"class": 9, "score": 0.95}
                        ],
                        "processing_time": 0.542
                    }
                }
            }
        },
        400: {
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [],
                        "error": "No file uploaded"
                    }
                }
            }
        },
        500: {
            "description": "Server error",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [],
                        "error": "Model not loaded"
                    }
                }
            }
        }
    }
)
async def predict(file: UploadFile = File(..., description="Image file to upload")):
    """
    Predict the class of a Vietnamese food image using the YOLO model.
    Upload a food image file and get predictions.
    
    Returns a list of predictions with class IDs and confidence scores.
    """
    start_time = time.time()
    temp_file_path = None
    
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded, can't perform prediction")
        return JSONResponse(
            content={"predictions": [], "error": "Model not loaded"}, 
            status_code=500
        )
    
    try:
        # Check if file is provided
        if not file:
            logger.error("No file uploaded")
            return JSONResponse(
                content={"predictions": [], "error": "No file uploaded"}, 
                status_code=400
            )
            
        # Log file information
        logger.info(f"Predicting from uploaded file: {file.filename}, content-type: {file.content_type}")
        
        # Save the file to disk temporarily
        file_extension = os.path.splitext(file.filename)[1]
        temp_file_path = os.path.join(TEMP_DIR, f"upload_{int(time.time())}_{file.filename}")
        
        logger.info(f"Saving temporary file to: {temp_file_path}")
        
        # Read file content
        contents = await file.read()
        
        # Save to temp file
        with open(temp_file_path, "wb") as f:
            f.write(contents)
            
        logger.info(f"Temporary file saved, size: {os.path.getsize(temp_file_path)} bytes")
        
        # Run prediction on the saved file path
        logger.info(f"Running prediction on file: {temp_file_path}")
        results = model(temp_file_path)
        logger.info("Prediction completed successfully")
        
        # Process results
        predictions = []
        if results:
            for r in results:
                # Extract classification results
                if hasattr(r, 'probs') and r.probs is not None:
                    # Get top classes and their probabilities
                    top_indices = r.probs.top5
                    top_values = r.probs.top5conf.tolist()
                    
                    for i, idx in enumerate(top_indices):
                        predictions.append({
                            "class": int(idx),
                            "score": float(top_values[i])
                        })
                        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Prediction completed in {processing_time:.3f}s with {len(predictions)} results")
        
        return {
            "predictions": predictions,
            "processing_time": processing_time
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"predictions": [], "error": str(e)}, 
            status_code=500
        )
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Deleted temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")

@app.post(
    "/predict-url", 
    summary="Predict food class from image URL", 
    description="Provide an image URL to classify Vietnamese food",
    response_model=PredictionResponse,
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {"class": 9, "score": 0.95}
                        ],
                        "processing_time": 0.542
                    }
                }
            }
        },
        400: {
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [],
                        "error": "Invalid URL"
                    }
                }
            }
        },
        500: {
            "description": "Server error",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [],
                        "error": "Model not loaded"
                    }
                }
            }
        }
    }
)
async def predict_url(url: str = Form(..., description="URL of the image to analyze")):
    """
    Predict Vietnamese food categories from an image URL.
    
    Args:
        url (str): URL of the image to analyze
        
    Returns:
        Prediction results with confidence scores
    """
    start_time = time.time()
    temp_file_path = None
    
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded, can't perform prediction")
        return JSONResponse(
            content={"predictions": [], "error": "Model not loaded"}, 
            status_code=500
        )
    
    try:
        # Check if URL is provided
        if not url:
            logger.error("No URL provided")
            return JSONResponse(
                content={"predictions": [], "error": "No URL provided"}, 
                status_code=400
            )
            
        logger.info(f"Predicting from URL: {url[:100]}...")
        
        # Download image from URL
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                logger.error(f"Could not download image from URL: {url}, status code: {response.status_code}")
                return JSONResponse(
                    content={"predictions": [], "error": "Could not download image"}, 
                    status_code=404
                )
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(response.content)
            temp_file.close()
            temp_file_path = temp_file.name
            
            logger.info(f"Downloaded image to temporary file: {temp_file_path}")
            
            # Run prediction on the saved file path
            results = model(temp_file_path)
            logger.info("Prediction completed successfully")
            
            # Process results
            predictions = []
            if results:
                for r in results:
                    # Extract classification results
                    if hasattr(r, 'probs') and r.probs is not None:
                        # Get top classes and their probabilities
                        top_indices = r.probs.top5
                        top_values = r.probs.top5conf.tolist()
                        
                        for i, idx in enumerate(top_indices):
                            predictions.append({
                                "class": int(idx),
                                "score": float(top_values[i])
                            })
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Prediction from URL completed in {processing_time:.3f}s with {len(predictions)} results")
            
            return {
                "predictions": predictions,
                "processing_time": processing_time
            }
    
    except Exception as e:
        logger.error(f"Error during prediction from URL: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"predictions": [], "error": str(e)}, 
            status_code=500
        )
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Deleted temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")

async def process_single_url(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """Process a single URL and return prediction results."""
    temp_file_path = None
    url_start_time = time.time()
    
    try:
        logger.info(f"Processing URL: {url[:100]}...")
        
        # Download image from URL with timeout
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=timeout)
            except httpx.TimeoutException:
                logger.error(f"Timeout downloading image from URL: {url}")
                return {
                    "success": False,
                    "error": "Timeout downloading image",
                    "url": url,
                    "processing_time": time.time() - url_start_time
                }
                
            if response.status_code != 200:
                logger.error(f"Could not download image from URL: {url}, status code: {response.status_code}")
                return {
                    "success": False,
                    "error": f"Could not download image, status code: {response.status_code}",
                    "url": url,
                    "processing_time": time.time() - url_start_time
                }
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(response.content)
            temp_file.close()
            temp_file_path = temp_file.name
            
            logger.info(f"Downloaded image to temporary file: {temp_file_path}")
            
            # Run prediction on the saved file path
            results = model(temp_file_path)
            
            # Process results
            predictions = []
            if results:
                for r in results:
                    # Extract classification results
                    if hasattr(r, 'probs') and r.probs is not None:
                        # Get top classes and their probabilities
                        top_indices = r.probs.top5
                        top_values = r.probs.top5conf.tolist()
                        
                        for i, idx in enumerate(top_indices):
                            predictions.append({
                                "class": int(idx),
                                "score": float(top_values[i])
                            })
            
            url_processing_time = time.time() - url_start_time
            logger.info(f"Processed URL {url[:30]}... in {url_processing_time:.3f}s with {len(predictions)} predictions")
            
            return {
                "success": True,
                "predictions": predictions,
                "url": url,
                "processing_time": url_processing_time
            }
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "processing_time": time.time() - url_start_time
        }
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Error deleting temporary file: {str(e)}")

@app.post(
    "/predict-batch",
    summary="Predict food classes from multiple image URLs",
    description="Process a batch of image URLs in parallel",
    response_model=BatchPredictionResponse,
    responses={
        200: {
            "description": "Successful batch prediction",
            "content": {
                "application/json": {
                    "example": {
                        "results": {
                            "http://example.com/image1.jpg": {
                                "success": True,
                                "predictions": [{"class": 9, "score": 0.95}],
                                "processing_time": 0.54
                            },
                            "http://example.com/image2.jpg": {
                                "success": True,
                                "predictions": [{"class": 5, "score": 0.87}],
                                "processing_time": 0.62
                            }
                        },
                        "processing_time": 1.2,
                        "success_count": 2,
                        "error_count": 0
                    }
                }
            }
        },
        400: {
            "description": "Bad request",
            "content": {
                "application/json": {
                    "example": {
                        "error": "No URLs provided"
                    }
                }
            }
        },
        500: {
            "description": "Server error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Model not loaded"
                    }
                }
            }
        }
    }
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Process multiple image URLs in parallel and return predictions for each.
    
    Args:
        request: BatchPredictionRequest containing URLs and optional timeout
        
    Returns:
        Dictionary mapping each URL to its prediction results
    """
    start_time = time.time()
    
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded, can't perform batch prediction")
        return JSONResponse(
            content={"error": "Model not loaded"}, 
            status_code=500
        )
    
    # Check if URLs are provided
    if not request.urls:
        logger.error("No URLs provided for batch prediction")
        return JSONResponse(
            content={"error": "No URLs provided"}, 
            status_code=400
        )
    
    logger.info(f"Received batch prediction request with {len(request.urls)} URLs")
    
    # Process URLs in parallel
    tasks = []
    for url in request.urls:
        if url:  # Skip empty URLs
            tasks.append(process_single_url(url, request.timeout_per_image))
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Organize results by URL
    result_dict = {}
    success_count = 0
    error_count = 0
    
    for result in results:
        url = result.get("url")
        if url:
            result_dict[url] = result
            if result.get("success", False):
                success_count += 1
            else:
                error_count += 1
    
    total_time = time.time() - start_time
    logger.info(f"Batch processing completed in {total_time:.3f}s. Success: {success_count}, Errors: {error_count}")
    
    return {
        "results": result_dict,
        "processing_time": total_time,
        "success_count": success_count,
        "error_count": error_count
    }

# Customize OpenAPI specification
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="YOLO Vietnamese Food Classification API",
        version="1.0.0",
        description="API for classifying Vietnamese food images using YOLOv8 model.",
        routes=app.routes,
    )
    
    # Add tag descriptions
    openapi_schema["tags"] = [
        {
            "name": "classification",
            "description": "Operations related to food classification",
        },
    ]
    
    # Add examples for the predict endpoint
    for path in openapi_schema["paths"]:
        if path in ["/predict", "/predict-url", "/predict-batch"]:
            openapi_schema["paths"][path]["post"]["tags"] = ["classification"]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    logger.info("Starting YOLO microservice")
    logger.info(f"Swagger UI will be available at http://localhost:8000/docs")
    logger.info(f"ReDoc will be available at http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)