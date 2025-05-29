import numpy as np
from typing import List, Dict, Any, Tuple
import cv2
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        self.max_history_length = 30
        self.next_id = 0
        self.tracked_objects: Dict[int, Dict[str, Any]] = {}

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        if not detections:
            return []

        # Update existing tracks and create new ones
        current_tracks = {}
        
        for det in detections:
            bbox = det['bbox']
            matched = False
            
            # Try to match with existing tracks
            for track_id, track in self.tracked_objects.items():
                iou = self._calculate_iou(bbox, track['bbox'])
                if iou > self.track_thresh:
                    # Update existing track
                    current_tracks[track_id] = {
                        'bbox': bbox,
                        'confidence': det['confidence'],
                        'class_id': det.get('class_id', 0),
                        'attributes': det.get('attributes', {})
                    }
                    
                    # Update track history
                    center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append(center)
                    
                    if len(self.track_history[track_id]) > self.max_history_length:
                        self.track_history[track_id].pop(0)
                    
                    matched = True
                    break
            
            # Create new track if no match found
            if not matched:
                track_id = self.next_id
                self.next_id += 1
                current_tracks[track_id] = {
                    'bbox': bbox,
                    'confidence': det['confidence'],
                    'class_id': det.get('class_id', 0),
                    'attributes': det.get('attributes', {})
                }
                
                # Initialize track history
                center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                self.track_history[track_id] = [center]

        # Update tracked objects
        self.tracked_objects = current_tracks

        # Convert to output format
        tracked_detections = []
        for track_id, track in current_tracks.items():
            tracked_det = {
                'track_id': track_id,
                'bbox': track['bbox'],
                'confidence': track['confidence'],
                'class_id': track['class_id'],
                'attributes': track['attributes'],
                'track_history': self.track_history.get(track_id, [])
            }
            tracked_detections.append(tracked_det)

        return tracked_detections

    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        return self.track_history.get(track_id, [])

    def clear_history(self):
        self.track_history.clear()
        self.tracked_objects.clear()
        self.next_id = 0
