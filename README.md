<p align="center">
  <img src="assets/CrowdInsight.png" alt="CrowdInsight Logo" width="500"/>
</p>

# CrowdInsight

CrowdInsight is a Python-based computer vision and AI library designed to analyze foot traffic and visitor demographics using CCTV footage or live streams. It detects and classifies entities such as adults, children, men, women, and animals (e.g., dogs), and generates actionable insights for businesses and institutions.

---

## 🚀 Features

- 📹 CCTV Video File & Live Stream Input  
- 🧠 Object Detection & Classification (adult, child, male, female, dog)  
- 🔁 Visitor Tracking with DeepSORT or ByteTrack  
- 📊 Analytics Engine for temporal and demographic patterns  
- 📤 Export to CSV/JSON for external dashboards  
- ⚙️ Modular and extendable  

---

## 📦 Installation

```bash
pip install crowdinsight
```

> Note: Currently under development – not yet on PyPI.

---

## 📁 Example Usage

### ▶️ Analyze a Video File

```python
from crowdinsight import CrowdAnalyzer

analyzer = CrowdAnalyzer(video_path="videos/cctv.mp4")
results = analyzer.run_analysis()

print(results.summary())
results.export_csv("output.csv")
```

### 📡 Analyze Live Webcam or IP Camera

```python
from crowdinsight import CrowdAnalyzer

analyzer = CrowdAnalyzer(video_source=0)  # Use 0 for default webcam or 'rtsp://...' for IP cam
analyzer.run_live_stream()
```

---

## 🧠 Model Backends

- YOLOv8 for real-time object detection  
- DeepSORT or ByteTrack for tracking across frames  
- Age & Gender Estimation with pre-trained CNNs  

---

## 💼 Use Cases

- 🏪 Retail: Analyze customer footfall & peak times  
- 🏨 Hotels: Understand demographics & optimize service  
- 🏛️ Government: Monitor public space usage  
- 🐶 Smart Cities: Pet & demographic-aware surveillance  

---

## 📊 Sample Output Format

```json
{
  "timestamp": "2025-05-27T14:45:12",
  "summary": {
    "total_visitors": 57,
    "adults": 39,
    "children": 10,
    "males": 29,
    "females": 24,
    "dogs": 4
  },
  "hourly_breakdown": {
    "09:00": {"visitors": 10},
    "10:00": {"visitors": 18}
  }
}
```

---

## 🛠️ Roadmap

- [ ] Video file support  
- [ ] Live stream camera support  
- [ ] Anomaly detection (e.g., crowd spikes)  
- [ ] Real-time dashboard  
- [ ] REST API with FastAPI  

---

## 🤝 Contributing

We welcome PRs and ideas! Feel free to open issues or contact us.

---

## 📄 License

MIT License

---

> Built with ❤️ by people who care about data-driven environments.
