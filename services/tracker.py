# ============================================================================
# services/tracker.py - ByteTrack Object Tracking
# ============================================================================

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from services.detector import Detection

@dataclass
class Track:
    track_id: int
    detections: List[Tuple[int, Detection, np.ndarray]] = field(default_factory=list)  # (frame_num, detection, image)
    last_seen: int = 0
    
    def add_detection(self, frame_num: int, detection: Detection, frame: np.ndarray):
        self.detections.append((frame_num, detection, frame))
        self.last_seen = frame_num
    
    def get_best_crop(self) -> Tuple[int, np.ndarray, float]:
        """Get the sharpest crop from all detections"""
        best_score = -1
        best_frame_num = 0
        best_crop = None
        
        for frame_num, det, frame in self.detections:
            x1, y1, x2, y2 = det.bbox
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Calculate sharpness using Laplacian variance
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if sharpness > best_score:
                best_score = sharpness
                best_frame_num = frame_num
                best_crop = crop
        
        return best_frame_num, best_crop, best_score


class SimpleTracker:
    """
    Simple IoU-based tracker for bottles on conveyor belt.
    For production, consider using ByteTrack or DeepSORT.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age  # Frames before track is considered lost
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, frame_num: int, detections: List[Detection], frame: np.ndarray) -> Dict[int, Detection]:
        """Update tracks with new detections"""
        
        # Remove old tracks
        expired = [tid for tid, t in self.tracks.items() 
                   if frame_num - t.last_seen > self.max_age]
        for tid in expired:
            del self.tracks[tid]
        
        # Match detections to existing tracks
        matched_detections = {}
        unmatched_detections = list(detections)
        
        for track_id, track in self.tracks.items():
            if not track.detections:
                continue
            
            last_detection = track.detections[-1][1]
            best_iou = 0
            best_det_idx = -1
            
            for i, det in enumerate(unmatched_detections):
                iou = self._calculate_iou(last_detection.bbox, det.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx >= 0:
                matched_det = unmatched_detections.pop(best_det_idx)
                track.add_detection(frame_num, matched_det, frame)
                matched_detections[track_id] = matched_det
        
        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            track = Track(track_id=self.next_track_id)
            track.add_detection(frame_num, det, frame)
            self.tracks[self.next_track_id] = track
            matched_detections[self.next_track_id] = det
            self.next_track_id += 1
        
        return matched_detections
    
    def get_all_tracks(self) -> List[Track]:
        """Get all tracks (including completed ones)"""
        return list(self.tracks.values())


