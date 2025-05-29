import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import colorsys

class Visualizer:
    def __init__(self):
        self.colors = self._generate_colors(100)  # Generate colors for up to 100 tracks
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
        return colors

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        for det in detections:
            bbox = det['bbox']
            label = det['label']
            confidence = det.get('confidence', 0.0)
            attributes = det.get('attributes', {})

            # Get color and category based on object type
            if label in ["person", "man", "woman", "boy", "girl", "child"]:
                color = (0, 255, 0)  # Green for people
                category = "Person"
            elif label in ["dog", "cat", "bird", "horse", "sheep", "cow"]:
                color = (0, 165, 255)  # Orange for animals
                category = "Animal"
            elif label in ["car", "bicycle", "motorcycle", "bus", "truck", "boat"]:
                color = (255, 0, 0)  # Blue for vehicles
                category = "Vehicle"
            else:
                color = (255, 255, 0)  # Cyan for other objects
                category = "Object"

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

            # Prepare label
            label_parts = [f"{category} ({confidence:.2f})"]
            
            # Add attributes based on object type
            if category == "Person":
                if attributes.get('gender'):
                    label_parts.append(f"Gender: {attributes['gender']}")
                if attributes.get('age_group'):
                    label_parts.append(f"Age: {attributes['age_group']}")
                if attributes.get('emotion'):
                    label_parts.append(f"Emotion: {attributes['emotion']}")
                if attributes.get('posture'):
                    label_parts.append(f"Posture: {attributes['posture']}")
            
            elif category == "Animal":
                label_parts.append(f"Type: {label}")
            
            elif category == "Vehicle":
                label_parts.append(f"Type: {label}")
            
            elif category == "Object":
                label_parts.append(f"Type: {label}")

            label = " | ".join(label_parts)

            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.thickness)

        return frame

    def draw_tracks(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id < 0:
                continue

            track_history = det.get('track_history', [])
            if not track_history:
                continue

            color = self.colors[track_id % len(self.colors)]

            # Draw track history
            for i in range(1, len(track_history)):
                cv2.line(frame, track_history[i-1], track_history[i], color, self.thickness)

        return frame

    def draw_stats(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Draw statistics on the frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw GPU info if available
        if "gpu_info" in stats:
            gpu_info = stats["gpu_info"]
            y_offset = 30
            cv2.putText(frame, f"GPU: {gpu_info['device']}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Memory: {gpu_info['memory_used']:.1f} GB", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 30
        
        # Draw statistics by category
        categories = {
            "People": ["total_visitors", "males", "females", "children", "adults"],
            "Animals": ["dogs", "cats", "birds"],
            "Vehicles": ["cars", "bicycles", "motorcycles"],
            "Objects": ["bottles", "cups", "phones", "laptops"]
        }
        
        y_offset = 80
        for category, items in categories.items():
            cv2.putText(frame, category, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 20
            
            for item in items:
                if item in stats:
                    value = stats[item]
                    if value > 0:  # Only show non-zero values
                        cv2.putText(frame, f"{item}: {value}", (30, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 20
        
        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        text = f"FPS: {fps:.1f}"
        # Create a semi-transparent background for FPS
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1] - 120, 10), (frame.shape[1] - 10, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, text, (frame.shape[1] - 110, 30), self.font, self.font_scale, (0, 255, 0), self.thickness)
        return frame
