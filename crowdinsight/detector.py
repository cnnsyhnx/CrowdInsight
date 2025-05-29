from ultralytics import YOLO
import cv2
from deepface import DeepFace
import numpy as np
from .config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, ALLOWED_CLASSES, DEVICE, AGE_GROUPS, POSTURE_THRESHOLDS
import torch
from typing import List, Dict, Any
import time

class ObjectDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD):
        try:
            # Force CUDA initialization
            if torch.cuda.is_available():
                # Set CUDA device
                torch.cuda.set_device(0)
                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
                # Get GPU info
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.cuda_available = True
                
                print("\n=== GPU Configuration ===")
                print(f"Device: {self.gpu_name}")
                print(f"Memory: {self.gpu_memory:.1f} GB")
                
                # Clear GPU memory
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()
                
                # Force CUDA memory allocation
                dummy_tensor = torch.zeros((1000, 1000), device='cuda')
                del dummy_tensor
                torch.cuda.synchronize()
            else:
                print("\n=== CPU Mode ===")
                self.cuda_available = False
                self.gpu_name = "CPU"
                self.gpu_memory = 0.0

            # Initialize YOLO model
            print(f"\nLoading YOLO model...")
            self.model = YOLO(model_path)
            
            # Force model to GPU if available
            if self.cuda_available:
                print("Moving model to GPU...")
                self.model = self.model.cuda()
                # Force CUDA memory allocation with model
                dummy_input = torch.zeros((1, 3, 640, 640), device='cuda')
                with torch.no_grad():
                    _ = self.model(dummy_input)
                torch.cuda.synchronize()
                print("Model moved to GPU successfully")
            
            self.conf_threshold = conf_threshold
            self.device = "cuda" if self.cuda_available else "cpu"
            
            # Initialize class categories
            self.person_classes = ["person", "man", "woman", "boy", "girl", "child"]
            self.animal_classes = ["dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
            self.vehicle_classes = ["car", "bicycle", "motorcycle", "bus", "truck", "boat"]
            self.food_classes = ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"]
            
            # Initialize face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.face_cache = {}
            self.cache_size = 100
            
            print("Initialization complete!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize detector: {str(e)}")

    def _detect_face(self, frame, bbox):
        """Detect face in the given bounding box and return face region"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            fx, fy, fw, fh = face
            return roi[fy:fy+fh, fx:fx+fw]
        return None

    def _analyze_attributes(self, frame: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
        try:
            x1, y1, x2, y2 = map(int, bbox)
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return {
                    "gender": "unknown",
                    "age_group": "unknown",
                    "posture": "unknown"
                }

            # Generate cache key
            cache_key = f"{x1}_{y1}_{x2}_{y2}"
            
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
                
                # Handle both single and multiple face results
                if isinstance(face_analysis, list):
                    face_analysis = face_analysis[0]

                # Get gender
                gender = face_analysis.get('gender', 'unknown')
                if isinstance(gender, dict):
                    gender = max(gender.items(), key=lambda x: x[1])[0].lower()
                elif isinstance(gender, str):
                    gender = gender.lower()
                attributes["gender"] = gender

                # Get age and determine age group
                age = face_analysis.get('age', 0)
                if age < 13:
                    age_group = "child"
                elif age < 60:
                    age_group = "adult"
                else:
                    age_group = "elderly"
                attributes["age_group"] = age_group

            except Exception as e:
                print(f"Warning: Face analysis failed: {str(e)}")
                # Fallback to RetinaFace if OpenCV fails
                try:
                    face_analysis = DeepFace.analyze(
                        face_img,
                        actions=['age', 'gender'],
                        enforce_detection=False,
                        detector_backend='retinaface',
                        silent=True
                    )
                    
                    # Process results as above
                    if isinstance(face_analysis, list):
                        face_analysis = face_analysis[0]
                    
                    # Update attributes with RetinaFace results
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
                    print(f"Warning: RetinaFace analysis also failed: {str(e)}")

            # Calculate aspect ratio for posture
            aspect_ratio = (x2 - x1) / (y2 - y1)
            attributes["posture"] = "standing" if aspect_ratio > 0.8 else "sitting"

            # Update cache
            if len(self.face_cache) >= self.cache_size:
                self.face_cache.pop(next(iter(self.face_cache)))
            self.face_cache[cache_key] = attributes

            return attributes

        except Exception as e:
            print(f"Warning: Face analysis error: {str(e)}")
            return {
                "gender": "unknown",
                "age_group": "unknown",
                "posture": "unknown"
            }

    def _get_gpu_utilization(self):
        """Get current GPU utilization"""
        if not self.cuda_available:
            return {
                "device": self.gpu_name,
                "memory_used": 0.0
            }
            
        try:
            # Force CUDA synchronization
            torch.cuda.synchronize()
            # Get memory info
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            return {
                "device": self.gpu_name,
                "memory_used": memory_allocated,
                "memory_reserved": memory_reserved
            }
        except Exception as e:
            print(f"Warning: GPU utilization tracking failed: {str(e)}")
            return {
                "device": self.gpu_name,
                "memory_used": 0.0,
                "memory_reserved": 0.0
            }

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU statistics"""
        return {
            "device": self.gpu_name,
            "total_memory": self.gpu_memory,
            "memory_used": self._get_gpu_utilization()["memory_used"]
        }

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        try:
            # Get GPU utilization before detection
            gpu_info = self._get_gpu_utilization()
            
            # Convert frame to tensor and move to GPU if available
            if self.cuda_available:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().cuda()
            
            # Run YOLO detection
            results = self.model(
                frame,
                verbose=False,
                conf=self.conf_threshold,
                device=self.device,
                half=True
            )
            
            # Convert generator to list and get first result
            results = list(results)[0]
            
            # Force CUDA synchronization
            if self.cuda_available:
                torch.cuda.synchronize()
            
            detections = []
            for box in results.boxes:
                confidence = float(box.conf[0])
                if confidence < self.conf_threshold:
                    continue

                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                if class_name not in ALLOWED_CLASSES:
                    continue

                bbox = box.xyxy[0].tolist()
                
                # Get attributes for person detections
                attributes = {}
                if class_name in self.person_classes:
                    attributes = self._analyze_attributes(frame, bbox)
                elif class_name in self.animal_classes:
                    attributes = {"type": class_name}
                elif class_name in self.vehicle_classes:
                    attributes = {"type": class_name}
                elif class_name in self.food_classes:
                    attributes = {"type": class_name}
                
                detections.append({
                    "bbox": bbox,
                    "label": class_name,
                    "confidence": confidence,
                    "attributes": attributes,
                    "gpu_info": gpu_info
                })

            return detections

        except Exception as e:
            print(f"Detection error: {str(e)}")
            return []

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


