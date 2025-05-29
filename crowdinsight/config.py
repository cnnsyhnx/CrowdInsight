YOLO_MODEL_PATH = "yolov8n.pt"  

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Allowed object classes for detection
ALLOWED_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]

# Device configuration
DEVICE = "cuda"  # Options: "cuda", "cpu"

# Face detection settings
FACE_DETECTION_MIN_SIZE = 30
FACE_DETECTION_SCALE_FACTOR = 1.1
FACE_DETECTION_MIN_NEIGHBORS = 5

# Attribute analysis settings
AGE_GROUPS = {
    "child": (0, 18),
    "adult": (18, 60),
    "elderly": (60, 100)
}

POSTURE_THRESHOLDS = {
    "standing": 2.5,
    "sitting": 1.5,
    "lying": 1.0
}

# Height and weight estimation
AVERAGE_PERSON_HEIGHT = 170  # cm
AVERAGE_BMI = 22

