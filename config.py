"""
Bottle Recycling Vision System - Backend
=========================================
A self-learning vision system for plastic bottle classification and impurity detection.

Project Structure:
------------------
bottle_vision/
├── main.py                 # FastAPI application entry point
├── config.py               # Configuration settings
├── database.py             # Database connection and models
├── models/
│   ├── __init__.py
│   ├── bottle.py           # Bottle-related database models
│   └── batch.py            # Batch-related database models
├── services/
│   ├── __init__.py
│   ├── video_processor.py  # Video frame extraction
│   ├── detector.py         # YOLO bottle detection
│   ├── tracker.py          # ByteTrack object tracking
│   ├── feature_extractor.py# Extract visual embeddings
│   ├── matcher.py          # Match bottles to known labels
│   └── batch_analyzer.py   # Batch analysis and reporting
├── api/
│   ├── __init__.py
│   ├── routes_batch.py     # Batch processing endpoints
│   ├── routes_labels.py    # Label management endpoints
│   └── routes_bottles.py   # Bottle management endpoints
└── utils/
    ├── __init__.py
    └── image_utils.py      # Image processing utilities

Requirements:
-------------
pip install fastapi uvicorn sqlalchemy python-multipart
pip install opencv-python numpy pillow
pip install torch torchvision
pip install ultralytics  # YOLOv8
pip install scikit-learn faiss-cpu
"""

# ============================================================================
# config.py - Configuration Settings
# ============================================================================

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/bottle_vision"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    CROPS_DIR: Path = BASE_DIR / "crops"
    MODELS_DIR: Path = BASE_DIR / "models_weights"
    
    # Video Processing
    FRAME_RATE: int = 5  # Extract 5 frames per second
    
    # Detection (YOLO)
    DETECTION_MODEL: str = "yolo11x.pt"
    YOLO_CONFIDENCE: float = 0.05
    
    # Tracking
    TRACK_BUFFER: int = 30  # Frames to keep track alive
    
    # Feature Extraction (ViT / CLIP)
    USE_CLIP: bool = False  # If True, use CLIP instead of ViT
    VIT_MODEL: str = "vit_l_32"  # Options: vit_b_16, vit_b_32, vit_l_16, vit_l_32
    CLIP_MODEL: str = "ViT-L/14@336px"  # Options: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
    
    # ViT embedding dimensions by model
    VIT_EMBEDDING_DIMS: dict = {
        "vit_b_16": 768,
        "vit_b_32": 768,
        "vit_l_16": 1024,
        "vit_l_32": 1024,
    }
    
    # Matching
    SIMILARITY_THRESHOLD: float = 0.01  # Primary threshold for matching
    HIGH_CONFIDENCE_THRESHOLD: float = 0.565
    MEDIUM_CONFIDENCE_THRESHOLD: float = 0.35
    
    # Clustering (HDBSCAN)
    HDBSCAN_MIN_CLUSTER_SIZE: int = 2
    HDBSCAN_MIN_SAMPLES: int = 1
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Create directories
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.CROPS_DIR.mkdir(exist_ok=True)
settings.MODELS_DIR.mkdir(exist_ok=True)


