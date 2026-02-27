# ============================================================================
# services/detector.py - YOLO Bottle Detection
# ============================================================================

from ultralytics import YOLO
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int

class BottleDetector:
    """Detect bottles in frames using YOLOv8"""
    
    # COCO class ID for bottle is 39
    BOTTLE_CLASS_ID = 39
    
    def __init__(self, model_path: str = "yolov11m.pt", confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect bottles in a single frame"""
        results = self.model(frame, conf=self.confidence, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            
            # Filter for bottles only (class 39 in COCO)
            # You can also train custom model for plastic bottles specifically
            if class_id == self.BOTTLE_CLASS_ID or True:  # Remove 'or True' after custom training
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=class_id
                ))
        
        return detections


