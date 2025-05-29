YOLO_MODEL_PATH = "yolov8n.pt"  

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.3

# Allowed object classes for detection
ALLOWED_CLASSES = [
    # People
    "person", "man", "woman", "boy", "girl", "child",
    
    # Animals
    "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    
    # Vehicles
    "car", "bicycle", "motorcycle", "bus", "truck", "boat",
    
    # Items
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket",
    
    # Food and Drinks
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    
    # Furniture
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    
    # Electronics
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    
    # Kitchen Items
    "microwave", "oven", "toaster", "sink", "refrigerator",
    
    # Other Items
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    
    # Additional Items
    "coffee cup", "mug", "glass", "plate", "bowl", "utensils"
]

# Device configuration
DEVICE = "cuda"  # Options: "cuda", "cpu"

# Face detection settings
FACE_DETECTION_MIN_SIZE = 20
FACE_DETECTION_SCALE_FACTOR = 1.05
FACE_DETECTION_MIN_NEIGHBORS = 3

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

