from ultralytics import YOLO
import cv2
from deepface import DeepFace
import numpy as np
from .config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, ALLOWED_CLASSES, DEVICE, AGE_GROUPS, POSTURE_THRESHOLDS
import torch
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

class ObjectDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        batch_size: int = 4,
        max_workers: int = 4
    ):
        self.logger = logging.getLogger("ObjectDetector")
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Initialize CUDA if available
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self._initialize_cuda()
        
        # Initialize YOLO model
        self._initialize_model(model_path)
        
        # Initialize face detection
        self._initialize_face_detection()
        
        # Initialize batch processing
        self.face_queue = Queue()
        self.result_queue = Queue()
        self._start_batch_processor()

    def _initialize_cuda(self) -> None:
        """Initialize CUDA settings and get GPU information"""
        try:
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            self.logger.info(f"GPU initialized: {self.gpu_name} ({self.gpu_memory:.1f} GB)")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CUDA: {str(e)}")
            self.cuda_available = False
            self.gpu_name = "CPU"
            self.gpu_memory = 0.0

    def _initialize_model(self, model_path: str) -> None:
        """Initialize YOLO model with error handling"""
        try:
            self.logger.info("Loading YOLO model...")
            self.model = YOLO(model_path)
            
            if self.cuda_available:
                self.model = self.model.cuda()
                # Warm up the model
                dummy_input = torch.zeros((1, 3, 640, 640), device='cuda')
                with torch.no_grad():
                    _ = self.model(dummy_input)
                torch.cuda.synchronize()
                self.logger.info("Model moved to GPU successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _initialize_face_detection(self) -> None:
        """Initialize face detection components"""
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.face_cache = {}
            self.cache_size = 100
            self.logger.info("Face detection initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize face detection: {str(e)}")
            raise RuntimeError(f"Face detection initialization failed: {str(e)}")

    def _start_batch_processor(self) -> None:
        """Start the batch processing thread for face detection"""
        self.batch_processor = threading.Thread(
            target=self._process_face_batches,
            daemon=True
        )
        self.batch_processor.start()

    def _process_face_batches(self) -> None:
        """Process batches of face detection requests"""
        while True:
            batch = []
            try:
                # Collect batch_size items or wait for timeout
                while len(batch) < self.batch_size:
                    try:
                        item = self.face_queue.get(timeout=0.1)
                        batch.append(item)
                    except Queue.Empty:
                        if batch:  # Process partial batch
                            break
                        continue

                if batch:
                    self._process_batch(batch)
            except Exception as e:
                self.logger.error(f"Batch processing error: {str(e)}")

    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of face detection requests"""
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for item in batch:
                    future = executor.submit(
                        self._analyze_face,
                        item['frame'],
                        item['bbox'],
                        item['cache_key']
                    )
                    futures.append((future, item['cache_key']))

                for future, cache_key in futures:
                    try:
                        result = future.result()
                        self.result_queue.put((cache_key, result))
                    except Exception as e:
                        self.logger.error(f"Face analysis error: {str(e)}")
                        self.result_queue.put((cache_key, None))

        except Exception as e:
            self.logger.error(f"Batch processing error: {str(e)}")

    def _analyze_face(
        self,
        frame: np.ndarray,
        bbox: List[int],
        cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single face with error handling"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return None

            # Check cache first
            if cache_key in self.face_cache:
                return self.face_cache[cache_key]

            # Initialize default attributes
            attributes = {
                "gender": "unknown",
                "age_group": "unknown",
                "posture": "unknown"
            }

            # Try with OpenCV first (faster)
            try:
                face_analysis = DeepFace.analyze(
                    face_img,
                    actions=['age', 'gender'],
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )
                
                if isinstance(face_analysis, list):
                    face_analysis = face_analysis[0]

                # Process gender
                gender = face_analysis.get('gender', 'unknown')
                if isinstance(gender, dict):
                    gender = max(gender.items(), key=lambda x: x[1])[0].lower()
                attributes["gender"] = gender

                # Process age
                age = face_analysis.get('age', 0)
                if age < 13:
                    age_group = "child"
                elif age < 60:
                    age_group = "adult"
                else:
                    age_group = "elderly"
                attributes["age_group"] = age_group

            except Exception as e:
                self.logger.warning(f"OpenCV face analysis failed: {str(e)}")
                # Fallback to RetinaFace
                try:
                    face_analysis = DeepFace.analyze(
                        face_img,
                        actions=['age', 'gender'],
                        enforce_detection=False,
                        detector_backend='retinaface',
                        silent=True
                    )
                    
                    if isinstance(face_analysis, list):
                        face_analysis = face_analysis[0]
                    
                    gender = face_analysis.get('gender', 'unknown')
                    if isinstance(gender, dict):
                        gender = max(gender.items(), key=lambda x: x[1])[0].lower()
                    attributes["gender"] = gender
                    
                    age = face_analysis.get('age', 0)
                    if age < 13:
                        age_group = "child"
                    elif age < 60:
                        age_group = "adult"
                    else:
                        age_group = "elderly"
                    attributes["age_group"] = age_group
                    
                except Exception as e:
                    self.logger.warning(f"RetinaFace analysis also failed: {str(e)}")

            # Calculate posture
            aspect_ratio = (x2 - x1) / (y2 - y1)
            attributes["posture"] = "standing" if aspect_ratio > 0.8 else "sitting"

            # Update cache
            if len(self.face_cache) >= self.cache_size:
                self.face_cache.pop(next(iter(self.face_cache)))
            self.face_cache[cache_key] = attributes

            return attributes

        except Exception as e:
            self.logger.error(f"Face analysis error: {str(e)}")
            return None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the frame with improved error handling"""
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.conf_threshold)[0]
            
            detections = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                if conf < self.conf_threshold:
                    continue
                
                bbox = [x1, y1, x2, y2]
                class_id = int(cls)
                label = results.names[class_id]
                
                # Queue face analysis for people
                if label in ["person", "man", "woman", "boy", "girl", "child"]:
                    cache_key = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
                    self.face_queue.put({
                        'frame': frame,
                        'bbox': bbox,
                        'cache_key': cache_key
                    })
                
                detection = {
                    'bbox': bbox,
                    'confidence': conf,
                    'class_id': class_id,
                    'label': label
                }
                
                detections.append(detection)
            
            # Process face analysis results
            while not self.result_queue.empty():
                cache_key, attributes = self.result_queue.get()
                if attributes:
                    for det in detections:
                        if det['label'] in ["person", "man", "woman", "boy", "girl", "child"]:
                            det['attributes'] = attributes
                            break
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {str(e)}")
            return []

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics with error handling"""
        try:
            if not self.cuda_available:
                return {
                    "device": self.gpu_name,
                    "total_memory": self.gpu_memory,
                    "memory_used": 0.0
                }
            
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            return {
                "device": self.gpu_name,
                "total_memory": self.gpu_memory,
                "memory_used": memory_allocated,
                "memory_reserved": memory_reserved
            }
            
        except Exception as e:
            self.logger.error(f"GPU stats error: {str(e)}")
            return {
                "device": self.gpu_name,
                "total_memory": self.gpu_memory,
                "memory_used": 0.0,
                "memory_reserved": 0.0
            }

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if self.cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def run_live_detection(self, camera_id=0, window_name="Live Detection"):
        """
        Run live object detection using a camera stream.
        
        Args:
            camera_id (int): Camera device ID (default: 0 for default camera)
            window_name (str): Name of the display window
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID: {camera_id}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Perform detection
                detections = self.detect(frame)

                # Draw detections on frame
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    label = det["label"]
                    conf = det["confidence"]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Prepare label text
                    label_text = f"{label} {conf:.2f}"
                    
                    # Add attributes if available
                    if "attributes" in det:
                        attrs = det["attributes"]
                        if attrs["gender"]:
                            label_text += f" | {attrs['gender']}"
                        if attrs["age_group"]:
                            label_text += f" | {attrs['age_group']}"
                        if attrs["posture"]:
                            label_text += f" | {attrs['posture']}"
                    
                    # Draw label background
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Display the frame
                cv2.imshow(window_name, frame)

                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


