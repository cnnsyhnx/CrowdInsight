import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import colorsys
import logging
from dataclasses import dataclass
from enum import Enum

class VisualizationMode(Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    MINIMAL = "minimal"

@dataclass
class VisualizationConfig:
    mode: VisualizationMode = VisualizationMode.DETAILED
    show_tracks: bool = True
    show_labels: bool = True
    show_stats: bool = True
    show_fps: bool = True
    track_history_length: int = 30
    label_font_scale: float = 0.5
    label_thickness: int = 2
    track_thickness: int = 2
    stats_font_scale: float = 0.5
    stats_thickness: int = 1
    overlay_alpha: float = 0.7

class Visualizer:
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.logger = logging.getLogger("Visualizer")
        self.config = config or VisualizationConfig()
        self.colors = self._generate_colors(100)  # Generate colors for up to 100 tracks
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self._initialize_colors()

    def _initialize_colors(self) -> None:
        """Initialize color schemes for different object types"""
        try:
            # Base colors for different categories
            self.category_colors = {
                "Person": (0, 255, 0),      # Green
                "Animal": (0, 165, 255),    # Orange
                "Vehicle": (255, 0, 0),     # Blue
                "Object": (255, 255, 0)     # Cyan
            }
            
            # Generate distinct colors for tracks
            self.track_colors = self._generate_colors(100)
            
            # Generate colors for different attributes
            self.attribute_colors = {
                "gender": {
                    "man": (0, 255, 0),     # Green
                    "woman": (0, 0, 255),   # Red
                    "unknown": (128, 128, 128)  # Gray
                },
                "age_group": {
                    "child": (255, 255, 0),  # Yellow
                    "adult": (0, 255, 255),  # Cyan
                    "elderly": (255, 0, 255), # Magenta
                    "unknown": (128, 128, 128) # Gray
                }
            }
        except Exception as e:
            self.logger.error(f"Color initialization error: {str(e)}")
            # Fallback to basic colors
            self.category_colors = {
                "Person": (0, 255, 0),
                "Animal": (0, 165, 255),
                "Vehicle": (255, 0, 0),
                "Object": (255, 255, 0)
            }
            self.track_colors = self._generate_colors(100)
            self.attribute_colors = {}

    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n visually distinct colors"""
        try:
            colors = []
            for i in range(n):
                hue = i / n
                saturation = 0.8
                value = 0.9
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
            return colors
        except Exception as e:
            self.logger.error(f"Color generation error: {str(e)}")
            return [(255, 0, 0)] * n  # Fallback to red

    def _get_category_color(self, label: str) -> Tuple[int, int, int]:
        """Get color for object category"""
        try:
            if label in ["person", "man", "woman", "boy", "girl", "child"]:
                return self.category_colors["Person"]
            elif label in ["dog", "cat", "bird", "horse", "sheep", "cow"]:
                return self.category_colors["Animal"]
            elif label in ["car", "bicycle", "motorcycle", "bus", "truck", "boat"]:
                return self.category_colors["Vehicle"]
            else:
                return self.category_colors["Object"]
        except Exception as e:
            self.logger.error(f"Category color error: {str(e)}")
            return (255, 255, 255)  # Fallback to white

    def _draw_label(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        label: str,
        color: Tuple[int, int, int]
    ) -> None:
        """Draw label with background"""
        try:
            # Get text size
            (label_width, label_height), _ = cv2.getTextSize(
                label,
                self.font,
                self.config.label_font_scale,
                self.config.label_thickness
            )
            
            # Draw background
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                self.font,
                self.config.label_font_scale,
                (255, 255, 255),
                self.config.label_thickness
            )
        except Exception as e:
            self.logger.error(f"Label drawing error: {str(e)}")

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detections on frame with improved visualization"""
        try:
            for det in detections:
                bbox = det['bbox']
                label = det['label']
                confidence = det.get('confidence', 0.0)
                attributes = det.get('attributes', {})
                track_id = det.get('track_id', -1)

                # Get color based on category
                color = self._get_category_color(label)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.label_thickness)

                if self.config.show_labels:
                    # Prepare label based on visualization mode
                    if self.config.mode == VisualizationMode.DETAILED:
                        label_parts = [f"{label} ({confidence:.2f})"]
                        if track_id >= 0:
                            label_parts.append(f"ID: {track_id}")
                        
                        # Add attributes based on object type
                        if label in ["person", "man", "woman", "boy", "girl", "child"]:
                            for attr, value in attributes.items():
                                if value != "unknown":
                                    label_parts.append(f"{attr}: {value}")
                    
                    elif self.config.mode == VisualizationMode.BASIC:
                        label_parts = [f"{label} ({confidence:.2f})"]
                        if track_id >= 0:
                            label_parts.append(f"ID: {track_id}")
                    
                    else:  # MINIMAL
                        label_parts = [f"{label}"]

                    label = " | ".join(label_parts)
                    self._draw_label(frame, x1, y1, label, color)

            return frame

        except Exception as e:
            self.logger.error(f"Detection drawing error: {str(e)}")
            return frame

    def draw_tracks(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw tracking history with improved visualization"""
        try:
            if not self.config.show_tracks:
                return frame

            for det in detections:
                track_id = det.get('track_id', -1)
                if track_id < 0:
                    continue

                track_history = det.get('track_history', [])
                if not track_history:
                    continue

                # Get color for track
                color = self.track_colors[track_id % len(self.track_colors)]

                # Draw track history
                for i in range(1, len(track_history)):
                    cv2.line(
                        frame,
                        track_history[i-1],
                        track_history[i],
                        color,
                        self.config.track_thickness
                    )

            return frame

        except Exception as e:
            self.logger.error(f"Track drawing error: {str(e)}")
            return frame

    def draw_stats(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Draw statistics with improved visualization"""
        try:
            if not self.config.show_stats:
                return frame

            # Create semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, self.config.overlay_alpha, frame, 1 - self.config.overlay_alpha, 0, frame)
            
            y_offset = 30
            
            # Draw GPU info if available
            if "gpu_info" in stats:
                gpu_info = stats["gpu_info"]
                cv2.putText(
                    frame,
                    f"GPU: {gpu_info['device']}",
                    (20, y_offset),
                    self.font,
                    self.config.stats_font_scale,
                    (255, 255, 255),
                    self.config.stats_thickness
                )
                y_offset += 20
                cv2.putText(
                    frame,
                    f"Memory: {gpu_info['memory_used']:.1f} GB",
                    (20, y_offset),
                    self.font,
                    self.config.stats_font_scale,
                    (255, 255, 255),
                    self.config.stats_thickness
                )
                y_offset += 30
            
            # Draw statistics by category
            categories = {
                "People": ["total_visitors", "males", "females", "children", "adults"],
                "Animals": ["dogs", "cats", "birds"],
                "Vehicles": ["cars", "bicycles", "motorcycles"],
                "Objects": ["bottles", "cups", "phones", "laptops"]
            }
            
            for category, items in categories.items():
                cv2.putText(
                    frame,
                    category,
                    (20, y_offset),
                    self.font,
                    self.config.stats_font_scale + 0.1,
                    (255, 255, 255),
                    self.config.stats_thickness
                )
                y_offset += 20
                
                for item in items:
                    if item in stats and stats[item] > 0:
                        cv2.putText(
                            frame,
                            f"{item}: {stats[item]}",
                            (30, y_offset),
                            self.font,
                            self.config.stats_font_scale,
                            (255, 255, 255),
                            self.config.stats_thickness
                        )
                        y_offset += 20
            
            return frame

        except Exception as e:
            self.logger.error(f"Stats drawing error: {str(e)}")
            return frame

    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """Draw FPS counter with improved visualization"""
        try:
            if not self.config.show_fps:
                return frame

            text = f"FPS: {fps:.1f}"
            
            # Create semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (frame.shape[1] - 120, 10),
                (frame.shape[1] - 10, 40),
                (0, 0, 0),
                -1
            )
            cv2.addWeighted(overlay, self.config.overlay_alpha, frame, 1 - self.config.overlay_alpha, 0, frame)
            
            # Draw FPS text
            cv2.putText(
                frame,
                text,
                (frame.shape[1] - 110, 30),
                self.font,
                self.config.stats_font_scale,
                (0, 255, 0),
                self.config.stats_thickness
            )
            
            return frame

        except Exception as e:
            self.logger.error(f"FPS drawing error: {str(e)}")
            return frame

    def __del__(self) -> None:
        """Cleanup resources"""
        try:
            # Clear any cached resources
            self.colors.clear()
            self.category_colors.clear()
            self.track_colors.clear()
            self.attribute_colors.clear()
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
