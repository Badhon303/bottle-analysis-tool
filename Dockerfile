# Use the slim version
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
# We added libxcb1 and other X11 libraries here
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# The rest of your steps remain the same
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8066
CMD ["python", "main.py"]