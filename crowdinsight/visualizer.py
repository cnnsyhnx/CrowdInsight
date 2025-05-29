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

            # Get color based on track_id
            color = self.colors[track_id % len(self.colors)] if track_id >= 0 else (0, 255, 0)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

            # Draw label
            label = f"{class_name} {confidence:.2f}"
            if track_id >= 0:
                label = f"ID:{track_id} {label}"

            (label_width, label_height), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
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
