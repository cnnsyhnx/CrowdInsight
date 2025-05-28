import unittest
import cv2
import os

from crowdinsight.detector import ObjectDetector
from crowdinsight.config import YOLO_MODEL_PATH


class TestObjectDetector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = ObjectDetector(model_path=YOLO_MODEL_PATH)

    def test_model_loads(self):
        self.assertIsNotNone(self.detector.model)

    def test_detect_on_sample_frame(self):
        frame = cv2.imread("videos/cctv.mp4")  
        if frame is None:
            frame = cv2.imread("assets/sample.jpg")  
        if frame is None:
            frame = cv2.UMat(480, 640, cv2.CV_8UC3).get()

        results = self.detector.detect(frame)
        self.assertIsInstance(results, list)
        for item in results:
            self.assertIn('class', item)
            self.assertIn('confidence', item)
            self.assertIn('bbox', item)


if __name__ == "__main__":
    unittest.main()
