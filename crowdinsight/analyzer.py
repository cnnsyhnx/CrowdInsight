import cv2
import time
from typing import Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

from .detector import ObjectDetector
from .tracker import ObjectTracker
from .visualizer import Visualizer
from .utils import setup_logger, save_results, calculate_demographics, format_results

class CrowdAnalyzer:
    def __init__(
        self,
        video_source: Union[str, int],
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        track_threshold: float = 0.5,
        show_video: bool = True
    ):
        self.logger = setup_logger("CrowdAnalyzer")
        self.video_source = video_source
        self.show_video = show_video
        
        # Initialize components
        self.detector = ObjectDetector(model_path, conf_threshold)
        self.tracker = ObjectTracker(track_threshold)
        self.visualizer = Visualizer()
        
        # Statistics
        self.stats = {
            "total_visitors": 0,
            "adults": 0,
            "children": 0,
            "males": 0,
            "females": 0,
            "dogs": 0
        }
        
        self.hourly_data = {}
        self.current_hour = None

    def _update_hourly_stats(self, detections: list):
        current_hour = time.strftime("%H:00")
        
        if current_hour != self.current_hour:
            self.current_hour = current_hour
            self.hourly_data[current_hour] = {"visitors": 0}
        
        self.hourly_data[current_hour]["visitors"] = len(detections)

    def _process_frame(self, frame: np.ndarray) -> tuple:
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Track objects
        tracked_detections = self.tracker.update(detections, frame)
        
        # Update statistics
        demographics = calculate_demographics(tracked_detections)
        self._update_hourly_stats(tracked_detections)
        
        # Update global stats
        for key in self.stats:
            self.stats[key] = max(self.stats[key], demographics[key])
        
        return tracked_detections, demographics

    def run_analysis(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {self.video_source}")

        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            tracked_detections, demographics = self._process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            if self.show_video:
                # Draw visualizations
                frame = self.visualizer.draw_detections(frame, tracked_detections)
                frame = self.visualizer.draw_tracks(frame, tracked_detections)
                frame = self.visualizer.draw_stats(frame, demographics)
                frame = self.visualizer.draw_fps(frame, fps)
                
                cv2.imshow("CrowdInsight", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.show_video:
            cv2.destroyAllWindows()

        # Prepare results
        results = format_results(self.stats, self.hourly_data)
        
        # Save results if output path is provided
        if output_path:
            save_results(results, output_path)
        
        return results

    def run_live_stream(self):
        return self.run_analysis()

    def export_results(self, output_path: str):
        results = format_results(self.stats, self.hourly_data)
        save_results(results, output_path)
        return results
