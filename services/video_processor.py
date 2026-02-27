"""
Services Layer - Core Processing Logic
======================================
"""

# ============================================================================
# services/video_processor.py - Video Frame Extraction
# ============================================================================

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple
from dataclasses import dataclass

@dataclass
class FrameData:
    frame_number: int
    timestamp: float
    image: np.ndarray

class VideoProcessor:
    """Extract frames from video at specified rate"""
    
    def __init__(self, video_path: str, target_fps: int = 5):
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        return self
    
    def __exit__(self, *args):
        if self.cap:
            self.cap.release()
    
    @property
    def video_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    @property
    def total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.video_fps if self.video_fps > 0 else 0
    
    def extract_frames(self) -> Generator[FrameData, None, None]:
        """Yield frames at target FPS"""
        frame_interval = int(self.video_fps / self.target_fps)
        frame_interval = max(1, frame_interval)
        
        frame_number = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_number % frame_interval == 0:
                timestamp = frame_number / self.video_fps
                yield FrameData(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    image=frame
                )
            
            frame_number += 1


