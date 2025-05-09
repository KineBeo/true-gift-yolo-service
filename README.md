# YOLO Vietnamese Food Classification Microservice

Microservice for Vietnamese food classification using YOLOv8.

## Features

- Fast food classification using YOLOv8
- Simple REST API for integration
- Supports image uploads and URL-based classification
- Batch processing support for multiple images
- OpenAPI documentation with Swagger UI and ReDoc

## API Endpoints

### Main Endpoints

- `GET /` - Basic health check
- `GET /status` - Service status and model information
- `POST /predict` - Upload and classify a single image
- `POST /predict-url` - Classify an image from a URL
- `POST /predict-batch` - Batch classify multiple images from URLs (parallel processing)

### Documentation

- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

## Examples

### Single Image Prediction

```bash
# Upload an image file
curl -X POST -F "file=@/path/to/your/image.jpg" http://localhost:8000/predict

# Classify from URL
curl -X POST -F "url=https://example.com/image.jpg" http://localhost:8000/predict-url
```

### Batch Prediction

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"], "timeout_per_image": 5.0}' \
  http://localhost:8000/predict-batch
```

## Response Format

### Single Image Response

```json
{
  "predictions": [
    {
      "class": 9,
      "score": 0.95
    },
    {
      "class": 12,
      "score": 0.03
    }
  ],
  "processing_time": 0.542
}
```

### Batch Prediction Response

```json
{
  "results": {
    "https://example.com/image1.jpg": {
      "success": true,
      "predictions": [
        {"class": 9, "score": 0.95}
      ],
      "url": "https://example.com/image1.jpg",
      "processing_time": 0.54
    },
    "https://example.com/image2.jpg": {
      "success": true,
      "predictions": [
        {"class": 5, "score": 0.87}
      ],
      "url": "https://example.com/image2.jpg",
      "processing_time": 0.62
    }
  },
  "processing_time": 1.2,
  "success_count": 2,
  "error_count": 0
}
```

## Running the Service

### With Docker

```bash
# Build the Docker image
docker build -t yolo-food-api .

# Run the Docker container
docker run -p 8000:8000 yolo-food-api
```

### Without Docker

```bash
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

## Development

### Requirements

- Python 3.8+
- FastAPI
- Ultralytics YOLOv8
- httpx (for URL based classification)

### Model

The service uses a YOLOv8 model trained on Vietnamese food images. The model file should be placed in the `data` directory with the name `yolov8-vn-food-classification.pt`.

### Environment Variables

- `PORT` - Port to run the service on (default: 8000)
- `HOST` - Host to run the service on (default: 0.0.0.0)
- `MODEL_PATH` - Path to the YOLOv8 model (default: data/yolov8-vn-food-classification.pt)

## Class Mapping

The model returns class IDs which map to Vietnamese food names. Here are some examples:

- Class 0: Bánh mì
- Class 1: Phở
- Class 2: Bún chả
- Class 3: Cơm tấm
- Class 4: Bánh xèo

See the full mapping in the service documentation. 