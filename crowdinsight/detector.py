from ultralytics import YOLO
import cv2
from crowdinsight.config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, ALLOWED_CLASSES

class ObjectDetector:
    def __init__ (self):
        self.model: YOLO = YOLO(YOLO_MODEL_PATH) 
        