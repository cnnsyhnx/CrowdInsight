# CrowdInsight

CrowdInsight is a Python-based computer vision and AI library designed to analyze foot traffic and visitor demographics using CCTV footage. It detects and classifies entities such as adults, children, men, women, and animals (e.g., dogs), and generates actionable insights for businesses and institutions.

## 🚀 Features

- 📹 CCTV Video Input support  
- 🧠 Object Detection & Classification (adult, child, male, female, dog)  
- 🔁 Tracking Visitors using DeepSORT or ByteTrack  
- 📊 Analytics Engine for temporal and demographic patterns  
- 📤 Export to CSV/JSON for integration with external dashboards  
- ⚙️ Modular and easy to integrate  

## 📦 Installation

```
pip install crowdinsight
```

> Note: Currently under development – not yet on PyPI.

## 📁 Example Usage

```python
from crowdinsight import CrowdAnalyzer

analyzer = CrowdAnalyzer(video_path="cctv.mp4")
results = analyzer.run_analysis()

print(results.summary())
results.export_csv("output.csv")
```

## 🧠 Model Backends

- YOLOv8 for object detection  
- DeepSORT for object tracking  
- Age & Gender Estimation via pre-trained models  

## 💼 Use Cases

- Retail stores analyzing peak hours  
- Hotels and event centers tracking visitor demographics  
- Government institutions managing public spaces  
- Pet detection in public places  

## 📊 Output Format

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

## 🛠️ Roadmap

- [ ] Live camera stream support  
- [ ] Dashboard integration  
- [ ] Anomaly detection  
- [ ] REST API with FastAPI  

## 🤝 Contributing

We welcome contributions! Please open issues and pull requests to improve the library.

## 📄 License

MIT License

---

**Built with ❤️ for AI-driven spatial analytics.**
