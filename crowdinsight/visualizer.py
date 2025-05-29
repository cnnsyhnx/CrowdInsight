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
            track_id = det.get('track_id', -1)
            class_name = det.get('class', 'unknown')
            confidence = det.get('confidence', 0.0)
            metrics = det.get('metrics', {})
            attributes = det.get('attributes', {})

            # Get color based on track_id
            color = self.colors[track_id % len(self.colors)] if track_id >= 0 else (0, 255, 0)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

            # Prepare detailed label
            label_parts = []
            if track_id >= 0:
                label_parts.append(f"ID:{track_id}")
            label_parts.append(f"{class_name}")
            label_parts.append(f"Conf:{confidence:.2f}")
            
            # Add attributes to label based on object type
            if class_name in ["person", "man", "woman", "boy", "girl", "child"]:
                if attributes.get('age_group'):
                    label_parts.append(f"Age:{attributes['age_group']}")
                if attributes.get('posture'):
                    label_parts.append(f"Posture:{attributes['posture']}")
                if attributes.get('estimated_height'):
                    label_parts.append(f"H:{attributes['estimated_height']}m")
                if attributes.get('estimated_weight'):
                    label_parts.append(f"W:{attributes['estimated_weight']}kg")
            
            elif class_name in ["dog", "cat"]:
                if attributes.get('breed'):
                    label_parts.append(f"Breed:{attributes['breed']}")
                if attributes.get('size'):
                    label_parts.append(f"Size:{attributes['size']}")
                if attributes.get('color'):
                    label_parts.append(f"Color:{attributes['color']}")
            
            elif class_name in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                if attributes.get('type'):
                    label_parts.append(f"Type:{attributes['type']}")
                if attributes.get('color'):
                    label_parts.append(f"Color:{attributes['color']}")
                if attributes.get('size'):
                    label_parts.append(f"Size:{attributes['size']}")

            # Add size information if available
            if metrics:
                label_parts.append(f"Size:{int(metrics.get('width', 0))}x{int(metrics.get('height', 0))}")
                if 'center_point' in metrics:
                    center_x, center_y = metrics['center_point']
                    cv2.circle(frame, (center_x, center_y), 3, color, -1)

            label = " | ".join(label_parts)

            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), self.font, self.font_scale, (255, 255, 255), self.thickness)

            # Draw track history if available
            if 'track_history' in det and det['track_history']:
                history = det['track_history']
                for i in range(1, len(history)):
                    cv2.line(frame, history[i-1], history[i], color, 1)

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

    def draw_stats(self, frame: np.ndarray, stats: Dict[str, int]) -> np.ndarray:
        y_offset = 30
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset), self.font, self.font_scale, (255, 255, 255), self.thickness)
            y_offset += 20
        return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame, text, (frame.shape[1] - 100, 30), self.font, self.font_scale, (0, 255, 0), self.thickness)
        return frame
