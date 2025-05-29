from ultralytics import YOLO
import cv2
from deepface import DeepFace
import numpy as np
from .config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, ALLOWED_CLASSES, DEVICE
import torch

class ObjectDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        try:
            self.model = YOLO(model_path)
            # Set device based on availability
            if DEVICE == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"
            else:
                self.device = DEVICE
                
            # Initialize face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def _detect_face(self, frame, bbox):
        """Detect face in the given bounding box and return face region"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            fx, fy, fw, fh = face
            return roi[fy:fy+fh, fx:fx+fw]
        return None

    def _analyze_attributes(self, frame, bbox):
        """Analyze attributes of detected object"""
        attributes = {
            "gender": None,
            "age_group": None,
            "clothing": None,
            "posture": None,
            "estimated_height": None,
            "estimated_weight": None
        }
        
        # Detect face for person
        face = self._detect_face(frame, bbox)
        if face is not None and face.size > 0:
            try:
                # Analyze face attributes
                analysis = DeepFace.analyze(
                    face,
                    actions=['age', 'gender', 'emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(analysis, list):
                    analysis = analysis[0]
                
                # Update attributes
                attributes["gender"] = analysis.get("gender", None)
                age = analysis.get("age", None)
                if age is not None:
                    if age < 18:
                        attributes["age_group"] = "child"
                    elif age < 60:
                        attributes["age_group"] = "adult"
                    else:
                        attributes["age_group"] = "elderly"
                
            except Exception as e:
                print(f"Face analysis error: {str(e)}")
        
        # Estimate height and weight based on bbox
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        if height > 0:
            # Rough estimation (assuming average person height is 170cm)
            estimated_height = (height / frame.shape[0]) * 170
            attributes["estimated_height"] = round(estimated_height, 1)
            
            # Rough weight estimation (BMI formula)
            if estimated_height > 0:
                bmi = 22  # Average BMI
                estimated_weight = (estimated_height/100) ** 2 * bmi
                attributes["estimated_weight"] = round(estimated_weight, 1)
        
        # Determine posture
        if height > 0 and width > 0:
            aspect_ratio = height / width
            if aspect_ratio > 2.5:
                attributes["posture"] = "standing"
            elif aspect_ratio > 1.5:
                attributes["posture"] = "sitting"
            else:
                attributes["posture"] = "lying"
        
        return attributes

    def detect(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")

        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=CONFIDENCE_THRESHOLD,
            device=self.device
        )[0]
        
        detections = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            label = self.model.names[int(class_id)]

            # Skip if class not in allowed classes
            if label not in ALLOWED_CLASSES:
                continue

            bbox = [int(x1), int(y1), int(x2), int(y2)]

            detection = {
                "label": label,
                "confidence": float(confidence),
                "bbox": bbox,
                "metrics": {
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "area": float((x2 - x1) * (y2 - y1)),
                    "center_point": [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                }
            }

            # Analyze attributes for person
            if label == "person":
                detection["attributes"] = self._analyze_attributes(frame, bbox)

            detections.append(detection)

        return detections

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


