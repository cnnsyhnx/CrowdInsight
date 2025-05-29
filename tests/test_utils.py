import unittest
import json
import os
from pathlib import Path
from crowdinsight.utils import (
    setup_logger,
    load_config,
    save_results,
    get_timestamp,
    calculate_demographics,
    format_results
)

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_config = {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.5,
            "track_threshold": 0.5
        }
        self.test_config_path = "test_config.json"
        self.test_output_path = "test_output.json"
        
        # Create test config file
        with open(self.test_config_path, 'w') as f:
            json.dump(self.test_config, f)

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)

    def test_setup_logger(self):
        logger = setup_logger("test_logger")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger")
        self.assertEqual(logger.level, 20)  # INFO level

    def test_load_config(self):
        config = load_config(self.test_config_path)
        self.assertEqual(config, self.test_config)

    def test_save_results_json(self):
        test_results = {
            "timestamp": "2024-03-14T12:00:00",
            "summary": {
                "total_visitors": 10,
                "adults": 8,
                "children": 2
            }
        }
        
        save_results(test_results, self.test_output_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(self.test_output_path))
        
        # Check content
        with open(self.test_output_path, 'r') as f:
            saved_results = json.load(f)
        self.assertEqual(saved_results, test_results)

    def test_get_timestamp(self):
        timestamp = get_timestamp()
        self.assertIsInstance(timestamp, str)
        self.assertEqual(len(timestamp), 19)  # Format: YYYY-MM-DDTHH:MM:SS

    def test_calculate_demographics(self):
        test_detections = [
            {"class": "person", "age": "adult", "gender": "male"},
            {"class": "person", "age": "child", "gender": "female"},
            {"class": "dog"},
            {"class": "person", "age": "adult", "gender": "female"}
        ]
        
        demographics = calculate_demographics(test_detections)
        
        self.assertEqual(demographics["total_visitors"], 4)
        self.assertEqual(demographics["adults"], 2)
        self.assertEqual(demographics["children"], 1)
        self.assertEqual(demographics["males"], 1)
        self.assertEqual(demographics["females"], 2)
        self.assertEqual(demographics["dogs"], 1)

    def test_format_results(self):
        demographics = {
            "total_visitors": 10,
            "adults": 8,
            "children": 2,
            "males": 5,
            "females": 5,
            "dogs": 0
        }
        
        hourly_data = {
            "09:00": {"visitors": 5},
            "10:00": {"visitors": 5}
        }
        
        results = format_results(demographics, hourly_data)
        
        self.assertIn("timestamp", results)
        self.assertIn("summary", results)
        self.assertIn("hourly_breakdown", results)
        self.assertEqual(results["summary"], demographics)
        self.assertEqual(results["hourly_breakdown"], hourly_data)

if __name__ == "__main__":
    unittest.main() 