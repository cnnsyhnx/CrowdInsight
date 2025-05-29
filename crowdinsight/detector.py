from ultralytics import YOLO
import cv2
from deepface import DeepFace
import torch
import json
import os
from datetime import datetime
import threading
import queue
import time
import numpy as np
import sys

# Constants
YOLO_MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5
ALLOWED_CLASSES = ["person", "dog"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ObjectDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        try:
            if not torch.cuda.is_available():
                print("Error: GPU (CUDA) is not available. This program requires a GPU to run.")
                print("Please make sure you have a compatible NVIDIA GPU and CUDA installed.")
                sys.exit(1)
            self.device = "cuda"
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.frame_queue = queue.Queue(maxsize=2)
            self.output_queue = queue.Queue(maxsize=10)
            self.is_running = False
            self.tracked_objects = {}
            self.next_id = 0
            self.frame_count = 0
            self.fps = 0
            self.fps_start_time = time.time()
            self.output_file = None
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0

    def _update_tracked_objects(self, detections):
        current_objects = {}
        for det in detections:
            if det["label"] != "person":
                continue
            bbox = det["bbox"]
            matched = False
            for obj_id, obj in self.tracked_objects.items():
                iou = self._calculate_iou(bbox, obj["bbox"])
                if iou > 0.5:
                    current_objects[obj_id] = {
                        "bbox": bbox,
                        "label": det["label"],
                        "confidence": round(float(det["confidence"]), 2),
                        "attributes": det.get("attributes", {}),
                        "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    matched = True
                    break
            if not matched:
                current_objects[self.next_id] = {
                    "bbox": bbox,
                    "label": det["label"],
                    "confidence": round(float(det["confidence"]), 2),
                    "attributes": det.get("attributes", {}),
                    "last_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.next_id += 1
        self.tracked_objects = current_objects
        return current_objects

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
            if label not in ALLOWED_CLASSES:
                continue
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            detection = {
                "label": label,
                "confidence": float(confidence),
                "bbox": bbox
            }
            if label == "person":
                face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if face.size > 0 and face.shape[0] > 20 and face.shape[1] > 20:
                    try:
                        analysis = DeepFace.analyze(
                            face,
                            actions=['age', 'gender', 'emotion'],
                            enforce_detection=False
                        )
                        detection["attributes"] = {
                            "age": analysis[0]["age"],
                            "gender": analysis[0]["dominant_gender"],
                            "emotion": analysis[0]["dominant_emotion"]
                        }
                    except Exception as e:
                        detection["attributes"] = {"error": str(e)}
            detections.append(detection)
        return detections

    def _process_frames(self):
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1)
                detections = self.detect(frame)
                tracked_objects = self._update_tracked_objects(detections)
                output_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tracked_objects": tracked_objects
                }
                self.output_queue.put(output_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in frame processing: {str(e)}")

    def _save_outputs(self):
        while self.is_running:
            try:
                output_data = self.output_queue.get(timeout=1)
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving output: {str(e)}")

    def _update_fps(self):
        self.frame_count += 1
        if time.time() - self.fps_start_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_start_time = time.time()

    def run_live_detection(self, camera_id=0, window_name="Live Detection", output_dir="outputs"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, "live_output.json")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID: {camera_id}")
        resolutions = [
            (640, 480),
            (1280, 720),
            (320, 240)
        ]
        for width, height in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            ret, frame = cap.read()
            if ret and frame is not None:
                break
        else:
            raise RuntimeError("Failed to initialize camera with any resolution")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        self.is_running = True
        process_thread = threading.Thread(target=self._process_frames)
        save_thread = threading.Thread(target=self._save_outputs)
        process_thread.start()
        save_thread.start()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self._update_fps()
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                try:
                    output_data = self.output_queue.get_nowait()
                    tracked_objects = output_data["tracked_objects"]
                except queue.Empty:
                    continue
                for obj_id, obj in tracked_objects.items():
                    x1, y1, x2, y2 = obj["bbox"]
                    label = obj["label"]
                    conf = obj["confidence"]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                    label_text = f"ID:{obj_id} {label} {conf:.2f}"
                    if "attributes" in obj:
                        attrs = obj["attributes"]
                        if "error" not in attrs:
                            label_text += f" | Age: {attrs['age']} | {attrs['gender']} | {attrs['emotion']}"
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"FPS: {self.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.is_running = False
            process_thread.join()
            save_thread.join()
            cap.release()
            cv2.destroyAllWindows()
