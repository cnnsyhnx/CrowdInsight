import unittest
import cv2
import json
import os
import numpy as np
from crowdinsight.detector import ObjectDetector
from crowdinsight.config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, ALLOWED_CLASSES

SAMPLE_IMAGE = "images/image-1.png"
OUTPUT_PATH = "outputs/detection_output.json"

def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

class TestObjectDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = ObjectDetector(YOLO_MODEL_PATH)
        if not os.path.exists(SAMPLE_IMAGE):
            os.makedirs(os.path.dirname(SAMPLE_IMAGE), exist_ok=True)
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            cv2.imwrite(SAMPLE_IMAGE, test_image)

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.device)

    def test_invalid_frame(self):
        with self.assertRaises(ValueError):
            self.detector.detect(None)
        with self.assertRaises(ValueError):
            self.detector.detect(np.array([]))

    def test_detect_and_export_to_json(self):
        frame = cv2.imread(SAMPLE_IMAGE)
        self.assertIsNotNone(frame, f"Failed to load image: {SAMPLE_IMAGE}")
        detections = self.detector.detect(frame)
        self.assertIsInstance(detections, list)
        if detections:
            self.assertIn("label", detections[0])
            self.assertIn("confidence", detections[0])
            self.assertIn("bbox", detections[0])
            for det in detections:
                self.assertGreaterEqual(det["confidence"], CONFIDENCE_THRESHOLD)
            for det in detections:
                self.assertIn(det["label"], ALLOWED_CLASSES)
        clean_detections = convert_numpy(detections)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(clean_detections, f, indent=2)
        self.assertTrue(os.path.exists(OUTPUT_PATH))

if __name__ == "__main__":
    unittest.main()
