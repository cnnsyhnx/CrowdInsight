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
        self.detector = ObjectDetector(model_path=model_path, conf_threshold=conf_threshold)
        self.tracker = ObjectTracker()
        self.visualizer = Visualizer()
        
        # Statistics
        self.stats = {
            # People
            "total_visitors": 0,
            "males": 0,
            "females": 0,
            "children": 0,
            "adults": 0,
            
            # Animals
            "dogs": 0,
            "cats": 0,
            "birds": 0,
            
            # Vehicles
            "cars": 0,
            "bicycles": 0,
            "motorcycles": 0,
            
            # Objects
            "bottles": 0,
            "cups": 0,
            "phones": 0,
            "laptops": 0
        }
        
        self.hourly_data = {}
        self.current_hour = None
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 100  # Keep last 100 frame times

    def _update_hourly_stats(self, detections: list):
        current_hour = time.strftime("%H:00")
        
        if current_hour != self.current_hour:
            self.current_hour = current_hour
            self.hourly_data[current_hour] = {"visitors": 0}
        
        self.hourly_data[current_hour]["visitors"] = len(detections)

    def _process_frame(self, frame: np.ndarray) -> tuple:
        try:
            start_time = time.time()
            
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Update statistics
            for det in detections:
                label = det["label"]
                attributes = det.get("attributes", {})
                
                # Handle people
                if label in ["person", "man", "woman", "boy", "girl", "child"]:
                    self.stats["total_visitors"] = 1
                    if attributes.get("gender") == "man":
                        self.stats["males"] = 1
                    elif attributes.get("gender") == "woman":
                        self.stats["females"] = 1
                    if attributes.get("age_group") == "child":
                        self.stats["children"] = 1
                    elif attributes.get("age_group") == "adult":
                        self.stats["adults"] = 1
                
                # Handle animals
                elif label == "dog":
                    self.stats["dogs"] = 1
                elif label == "cat":
                    self.stats["cats"] = 1
                elif label == "bird":
                    self.stats["birds"] = 1
                
                # Handle vehicles
                elif label == "car":
                    self.stats["cars"] = 1
                elif label == "bicycle":
                    self.stats["bicycles"] = 1
                elif label == "motorcycle":
                    self.stats["motorcycles"] = 1
                
                # Handle objects
                elif label == "bottle":
                    self.stats["bottles"] = 1
                elif label == "cup":
                    self.stats["cups"] = 1
                elif label == "cell phone":
                    self.stats["phones"] = 1
                elif label == "laptop":
                    self.stats["laptops"] = 1
            
            # Calculate demographics
            demographics = {
                "People": {
                    "total_visitors": self.stats["total_visitors"],
                    "males": self.stats["males"],
                    "females": self.stats["females"],
                    "children": self.stats["children"],
                    "adults": self.stats["adults"]
                },
                "Animals": {
                    "dogs": self.stats["dogs"],
                    "cats": self.stats["cats"],
                    "birds": self.stats["birds"]
                },
                "Vehicles": {
                    "cars": self.stats["cars"],
                    "bicycles": self.stats["bicycles"],
                    "motorcycles": self.stats["motorcycles"]
                },
                "Objects": {
                    "bottles": self.stats["bottles"],
                    "cups": self.stats["cups"],
                    "phones": self.stats["phones"],
                    "laptops": self.stats["laptops"]
                }
            }
            
            # Track frame processing time
            frame_time = time.time() - start_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_times:
                self.frame_times.pop(0)
            
            return detections, demographics
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return [], {}

    def run_analysis(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {self.video_source}")

        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        start_time = time.time()
        frame_skip = 3  # Process every 3rd frame for better performance
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for better performance
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))

            # Process frame
            detections, demographics = self._process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            if self.show_video:
                # Get GPU stats
                gpu_stats = self.detector.get_gpu_stats()
                if gpu_stats:
                    self.stats["gpu_info"] = gpu_stats
                
                # Draw visualizations
                frame = self.visualizer.draw_detections(frame, detections)
                frame = self.visualizer.draw_stats(frame, self.stats)
                frame = self.visualizer.draw_fps(frame, fps)
                
                cv2.imshow("CrowdInsight", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.show_video:
            cv2.destroyAllWindows()

        # Calculate average frame processing time
        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        
        # Get final GPU stats
        gpu_stats = self.detector.get_gpu_stats()
        
        # Print analysis summary
        print("\n=== Analysis Summary ===")
        print(f"\nPerformance:")
        print(f"  Average Frame Time: {avg_frame_time*1000:.1f}ms")
        print(f"  Average FPS: {1/avg_frame_time:.1f}" if avg_frame_time > 0 else "  Average FPS: N/A")
        
        if gpu_stats:
            print(f"\nGPU Statistics:")
            print(f"  Device: {gpu_stats['device']}")
            print(f"  Total Memory: {gpu_stats['total_memory']:.1f} GB")
            print(f"  Memory Used: {gpu_stats['memory_used']:.1f} GB")
        
        for category, stats in demographics.items():
            print(f"\n{category}:")
            for key, value in stats.items():
                if key != "total" or value > 0:  # Only show total if it's greater than 0
                    if isinstance(value, dict):
                        print(f"  {key.capitalize()}:")
                        for subkey, subvalue in value.items():
                            if subvalue > 0:
                                print(f"    {subkey}: {subvalue}")
                    else:
                        print(f"  {key.capitalize()}: {value}")
        
        # Prepare results
        results = {
            "categories": demographics,
            "hourly_data": self.hourly_data,
            "performance": {
                "avg_frame_time": avg_frame_time,
                "avg_fps": 1/avg_frame_time if avg_frame_time > 0 else 0,
                "gpu_stats": gpu_stats
            }
        }
        
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
