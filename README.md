<p align="center">
  <img src="assets/CrowdInsight.png" alt="CrowdInsight Logo" width="1000px"/>
</p>

# CrowdInsight

**Know your crowd. Shape your space.**

CrowdInsight is a Python AI library that converts CCTV footage or live streams into meaningful insights. It detects and classifies entities like adults, children, men, women, and animals (e.g., dogs), tracks movement, and generates structured data for real-world applications — from retail stores to government buildings.

---

## 📁 Project Structure

```bash
CrowdInsight/
│
├── crowdinsight/             # Core Python package
│   ├── __init__.py
│   ├── analyzer.py           # Main interface (CrowdAnalyzer class)
│   ├── detector.py           # Object detection logic (YOLOv8, etc.)
│   ├── tracker.py            # Person tracking module (DeepSORT, ByteTrack)
│   ├── utils.py              # Helper functions
│   ├── visualizer.py         # Draw boxes, labels, etc.
│   └── config.py             # Configuration settings
│
├── examples/                 # Example scripts
│   ├── analyze_video.py
│   └── analyze_live.py
│
├── assets/                   # Branding, logos, sample media
│   └── CrowdInsight.png
│
├── videos/                   # Sample video files (for local tests)
│   └── cctv.mp4
│
├── outputs/                  # Output results (CSV, JSON, images)
│   └── output.csv
│
├── tests/                    # Unit tests
│   ├── test_analyzer.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   └── test_utils.py 
│
├── .gitignore                # Ignore files/folders for GitHub
├── LICENSE                   # Creative Commons Attribution-NoDerivatives 4.0 International Public License
├── README.md                 # Project overview and instructions
├── requirements.txt          # Required Python packages
└── setup.py                  # PyPI packaging file
```

---

## 🚀 Features

- 🎥 Video file & live webcam/IP stream support  
- 🧠 Real-time object detection and classification  
- 📍 Visitor tracking using DeepSORT/ByteTrack  
- 📊 Structured analytics export (CSV/JSON)  
- ⚙️ Modular and extensible for custom use cases  

---

## 📦 Installation

```bash
pip install crowdinsight
```

> ⚠️ Note: This project is currently in development and not yet on PyPI.

---

## 💡 Example Usage

### ▶️ Analyze a Video File (Python)

```python
from crowdinsight import CrowdAnalyzer

# Initialize the analyzer with a video file
analyzer = CrowdAnalyzer(
    video_source="videos/cctv.mp4",    # Video file path
    model_path="yolov8n.pt",           # YOLO model path
    conf_threshold=0.5,                # Detection confidence threshold
    show_video=True                    # Show real-time visualization (set False for headless)
)

# Run analysis and get results (results saved to output_path)
results = analyzer.run_analysis(output_path="outputs/results.json")

# Print all available analytics
print("\nAnalysis Summary:")
summary = results.get('summary', results.get('categories', {}))
for key, value in summary.items():
    print(f"{key.capitalize()}: {value}")

# Print hourly breakdown if available
hourly = results.get('hourly_breakdown', results.get('hourly_data', {}))
if hourly:
    print("\nHourly Breakdown:")
    for hour, data in hourly.items():
        visitors = data['visitors'] if isinstance(data, dict) and 'visitors' in data else data
        print(f"{hour}: {visitors} visitors")

print("\nResults saved to: outputs/results.json")
```

### 📡 Analyze a Live Stream (Python)

```python
from crowdinsight import CrowdAnalyzer

# Initialize with webcam (0) or IP camera
analyzer = CrowdAnalyzer(
    video_source=0,                    # 0 for webcam, or RTSP URL for IP camera
    model_path="yolov8n.pt",
    conf_threshold=0.5,
    show_video=True                    # Set False for headless servers
)

# Run live analysis
results = analyzer.run_live_stream()

# Export results if needed
analyzer.export_results("outputs/live_results.json")
```

### 🖥️ Command Line Usage

```bash
# For live stream analysis (with all options)
python examples/analyze_live.py --source 0 --model yolov8n.pt --conf 0.5 --output outputs/live_results.json --no-show

# For video file analysis (with all options)
python examples/analyze_video.py --video videos/cctv.mp4 --model yolov8n.pt --conf 0.5 --output outputs/results.json --no-show
```

> **Tip:** Use `--no-show` to disable video display for headless environments (e.g., servers, Docker).

### 📤 Exporting Results

Both the Python API and CLI examples support exporting analytics to JSON (or CSV, if implemented). Use the `output_path` argument or `--output` flag to specify the file.

### 📋 Requirements

```bash
# Core dependencies
opencv-python>=4.8.0
torch>=2.0.0
ultralytics>=8.0.0
numpy>=1.24.0
pandas>=2.0.0
filterpy>=1.4.5
deepface>=0.0.79
pillow>=10.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
python-logging>=0.4.9.6
```

### 🎯 Key Features

- **Real-time Visualization**:
  - Bounding boxes with object IDs
  - Tracking trails showing movement history
  - FPS counter
  - Current statistics display
  - Press 'q' to quit

- **Detection & Tracking**:
  - Person detection and classification
  - Age estimation (adults/children)
  - Gender estimation (male/female)
  - Animal detection (dogs)
  - Object tracking with unique IDs

- **Analytics**:
  - Real-time visitor counting
  - Demographic breakdown
  - Hourly statistics
  - Movement patterns
  - Export to JSON/CSV

---

## 🧠 Technology Stack

- YOLOv8 / YOLO-NAS (object detection)  
- DeepSORT / ByteTrack (object tracking)  
- Pre-trained CNNs (age/gender estimation)  
- OpenCV, PyTorch, Ultralytics  

---

## 📊 Sample Output (JSON)

```json
{
  "timestamp": "2025-05-27T15:00:00",
  "summary": {
    "total_visitors": 73,
    "adults": 49,
    "children": 12,
    "males": 37,
    "females": 28,
    "dogs": 5
  },
  "hourly_breakdown": {
    "09:00": {"visitors": 10},
    "10:00": {"visitors": 21},
    "11:00": {"visitors": 42}
  }
}
```

---

## 🧪 Testing

Run all unit tests:

```bash
pytest tests/
```

---

## 🗺️ Roadmap

- [x] Video file analysis  
- [ ] Live stream support  
- [ ] Basic demographic tracking  
- [ ] Real-time anomaly detection  
- [ ] Web dashboard integration (next stage)  
- [ ] REST API (FastAPI)  
- [ ] Deployment container (Docker)  

---

## 🎯 Target Use Cases

- 🏪 **Retail**: Understand customer flow and demographics  
- 🏨 **Hospitality**: Improve service based on real visitor data  
- 🏛️ **Public Sector**: Monitor and analyze usage of public spaces  
- 🐾 **Pet-Friendly Spaces**: Identify non-human traffic too  

---

## 🤝 Contributing

Got ideas? Found bugs? Want to collaborate?  
We welcome contributions!  
Please open an issue or submit a pull request.

---

## 📄 License

**CC BY-ND 4.0**  

CrowdInsight is licensed under the [Creative Commons Attribution-NoDerivatives 4.0 International License (CC BY-ND 4.0)](https://creativecommons.org/licenses/by-nd/4.0/).

![License: CC BY-ND 4.0](https://img.shields.io/badge/License-CC%20BY--ND%204.0-lightgrey.svg)

---

> Built with ❤️ to bridge the physical world and data intelligence.
