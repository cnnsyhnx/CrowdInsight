import unittest
import numpy as np
from crowdinsight import CrowdAnalyzer
from pathlib import Path
import os
import shutil

class TestCrowdAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = CrowdAnalyzer(
            video_source=0,  # Use webcam for testing
            model_path="yolov8n.pt",
            conf_threshold=0.5,
            show_video=False
        )

    def test_initialization(self):
        self.assertIsNotNone(self.analyzer.detector)
        self.assertIsNotNone(self.analyzer.tracker)
        self.assertIsNotNone(self.analyzer.visualizer)
        self.assertIsNotNone(self.analyzer.stats)

    def test_stats_initialization(self):
        expected_stats = {
            "total_visitors": 0,
            "adults": 0,
            "children": 0,
            "males": 0,
            "females": 0,
            "dogs": 0
        }
        self.assertEqual(self.analyzer.stats, expected_stats)

    def test_process_frame(self):
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        detections, demographics = self.analyzer._process_frame(frame)
        
        # Check return types
        self.assertIsInstance(detections, list)
        self.assertIsInstance(demographics, dict)
        
        # Check demographics structure
        expected_keys = ["total_visitors", "adults", "children", "males", "females", "dogs"]
        for key in expected_keys:
            self.assertIn(key, demographics)

    def test_export_results(self):
        # Create dummy results
        self.analyzer.stats = {
            "total_visitors": 10,
            "adults": 8,
            "children": 2,
            "males": 5,
            "females": 5,
            "dogs": 0
        }
        # Use a unique output directory for this test
        output_dir = "test_analyzer_outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_output.json")
        results = self.analyzer.export_results(output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertIn("timestamp", results)
        self.assertIn("summary", results)
        self.assertEqual(results["summary"], self.analyzer.stats)
        # Clean up
        try:
            os.remove(output_path)
            shutil.rmtree(output_dir)
        except Exception:
            pass

if __name__ == "__main__":
    unittest.main() 