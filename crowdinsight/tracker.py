import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import cv2
import logging
from filterpy.kalman import KalmanFilter
from collections import deque

class TrackedObject:
    def __init__(self, track_id: int, bbox: List[float], class_id: int, attributes: Dict[str, Any]):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.attributes = attributes
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.history = deque(maxlen=30)  # Store last 30 positions
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # State: [x, y, s, r, x', y', s'] where s is scale and r is aspect ratio
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R[2:, 2:] *= 10.
        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.kf.x[:4] = [x1 + w/2, y1 + h/2, w * h, w/h]
        
        # Initialize covariance
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.

    def predict(self) -> np.ndarray:
        """Predict next state using Kalman filter"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x

    def update(self, bbox: List[float]) -> None:
        """Update track with new detection"""
        self.time_since_update = 0
        self.hits += 1
        
        # Update Kalman filter
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        self.kf.update([x1 + w/2, y1 + h/2, w * h, w/h])
        
        # Update bbox
        self.bbox = bbox
        
        # Update history
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        self.history.append(center)

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the track"""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'class_id': self.class_id,
            'attributes': self.attributes,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'history': list(self.history)
        }

class ObjectTracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        max_age: int = 30,
        min_hits: int = 3
    ):
        self.logger = logging.getLogger("ObjectTracker")
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 0
        self.frame_count = 0

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        try:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection
            
            return intersection / union if union > 0 else 0
        except Exception as e:
            self.logger.error(f"IoU calculation error: {str(e)}")
            return 0.0

    def _match_detections_to_tracks(
        self,
        detections: List[Dict[str, Any]]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to existing tracks using Hungarian algorithm"""
        try:
            if not self.tracks or not detections:
                return [], list(range(len(detections))), list(self.tracks.keys())

            # Calculate cost matrix
            cost_matrix = np.zeros((len(detections), len(self.tracks)))
            for i, det in enumerate(detections):
                for j, track in enumerate(self.tracks.values()):
                    cost_matrix[i, j] = 1 - self._calculate_iou(det['bbox'], track.bbox)

            # Apply Hungarian algorithm
            row_ind, col_ind = cv2.linear_sum_assignment(cost_matrix)

            # Filter matches based on IoU threshold
            matches = []
            unmatched_detections = []
            unmatched_tracks = list(range(len(self.tracks)))

            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] < (1 - self.track_thresh):
                    matches.append((i, j))
                    unmatched_tracks.remove(j)
                else:
                    unmatched_detections.append(i)

            return matches, unmatched_detections, unmatched_tracks

        except Exception as e:
            self.logger.error(f"Detection matching error: {str(e)}")
            return [], list(range(len(detections))), list(range(len(self.tracks)))

    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """Update tracks with new detections"""
        try:
            self.frame_count += 1
            
            # Predict new locations of existing tracks
            for track in self.tracks.values():
                track.predict()

            # Match detections to existing tracks
            matches, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(detections)

            # Update matched tracks
            for det_idx, track_idx in matches:
                track_id = list(self.tracks.keys())[track_idx]
                self.tracks[track_id].update(detections[det_idx]['bbox'])

            # Create new tracks for unmatched detections
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                self.tracks[self.next_id] = TrackedObject(
                    self.next_id,
                    det['bbox'],
                    det.get('class_id', 0),
                    det.get('attributes', {})
                )
                self.next_id += 1

            # Remove old tracks
            self.tracks = {
                track_id: track for track_id, track in self.tracks.items()
                if track.time_since_update < self.max_age and track.hits >= self.min_hits
            }

            # Return updated tracks
            return [track.get_state() for track in self.tracks.values()]

        except Exception as e:
            self.logger.error(f"Track update error: {str(e)}")
            return []

    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """Get history of a specific track"""
        try:
            if track_id in self.tracks:
                return list(self.tracks[track_id].history)
            return []
        except Exception as e:
            self.logger.error(f"Track history error: {str(e)}")
            return []

    def clear_history(self) -> None:
        """Clear all tracking history"""
        try:
            self.tracks.clear()
            self.next_id = 0
            self.frame_count = 0
        except Exception as e:
            self.logger.error(f"History clear error: {str(e)}")

    def __del__(self) -> None:
        """Cleanup resources"""
        try:
            self.clear_history()
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
