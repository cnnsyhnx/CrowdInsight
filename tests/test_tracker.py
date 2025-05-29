import unittest
import numpy as np
from crowdinsight import ObjectTracker

class TestObjectTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjectTracker(track_thresh=0.5, track_buffer=30)

    def test_initialization(self):
        self.assertIsNotNone(self.tracker)
        self.assertEqual(len(self.tracker.track_history), 0)

    def test_update_with_empty_detections(self):
        # Create empty frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Update with empty detections
        tracked_detections = self.tracker.update([], frame)
        
        # Check result
        self.assertEqual(len(tracked_detections), 0)
        self.assertEqual(len(self.tracker.track_history), 0)

    def test_update_with_detections(self):
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create dummy detections
        detections = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.9,
                'class_id': 0
            },
            {
                'bbox': [300, 300, 400, 400],
                'confidence': 0.8,
                'class_id': 0
            }
        ]
        
        # Update tracker
        tracked_detections = self.tracker.update(detections, frame)
        
        # Check results
        self.assertIsInstance(tracked_detections, list)
        self.assertGreater(len(tracked_detections), 0)
        
        # Check track history
        self.assertGreater(len(self.tracker.track_history), 0)
        
        # Check detection format
        for det in tracked_detections:
            self.assertIn('track_id', det)
            self.assertIn('bbox', det)
            self.assertIn('confidence', det)
            self.assertIn('class_id', det)
            self.assertIn('track_history', det)

    def test_get_track_history(self):
        # Create dummy frame and detections
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [{
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9,
            'class_id': 0
        }]
        
        # Update tracker
        tracked_detections = self.tracker.update(detections, frame)
        
        if tracked_detections:
            track_id = tracked_detections[0]['track_id']
            history = self.tracker.get_track_history(track_id)
            
            # Check history
            self.assertIsInstance(history, list)
            self.assertGreater(len(history), 0)
            
            # Check history points
            for point in history:
                self.assertIsInstance(point, tuple)
                self.assertEqual(len(point), 2)

    def test_clear_history(self):
        # Create dummy frame and detections
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [{
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9,
            'class_id': 0
        }]
        
        # Update tracker
        self.tracker.update(detections, frame)
        
        # Clear history
        self.tracker.clear_history()
        
        # Check if history is cleared
        self.assertEqual(len(self.tracker.track_history), 0)

if __name__ == "__main__":
    unittest.main() 