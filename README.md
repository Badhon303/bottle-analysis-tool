# Bottle Vision System - Backend

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Installation


1.  **Create a virtual environment** (recommended to keep dependencies isolated):

    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Linux / macOS
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install the required dependencies**:

    ```bash
    pip install fastapi uvicorn sqlalchemy python-multipart opencv-python numpy pillow torch torchvision ultralytics scikit-learn faiss-cpu pydantic-settings python-dotenv
    ```

## Configuration

The application uses a `.env` file for configuration. By default, it uses a local SQLite database and predefined model paths.

You can create a `.env` file in the root directory to override defaults:

```ini
# Database
DATABASE_URL="sqlite:///./bottle_vision.db"

# YOLO Model
DETECTION_MODEL="yolo11x.pt"
YOLO_CONFIDENCE=0.05
```

## Running the Application

To start the server, run:

```bash
python main.py
```

> **Note**: The first time you run the application, it will download the YOLO model weights (e.g., `yolo11x.pt`) automatically. Please ensure you have an internet connection.

The server will start at `http://0.0.0.0:8066`.

- **API Documentation**: [http://localhost:8066/docs](http://localhost:8066/docs)
- **Alternative Documentation**: [http://localhost:8066/redoc](http://localhost:8066/redoc)
