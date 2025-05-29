import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results: Dict, output_path: str) -> None:
    """Save analysis results to a JSON file."""
    # Use a local logger
    logger = logging.getLogger("CrowdInsightUtils")
    try:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        try:
            logger.error(f"Error saving results: {str(e)}")
        except Exception:
            pass
        raise

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

def calculate_demographics(detections: list) -> Dict[str, int]:
    demographics = {
        "total_visitors": len(detections),
        "adults": 0,
        "children": 0,
        "males": 0,
        "females": 0,
        "dogs": 0
    }
    
    for detection in detections:
        if detection.get("class") == "dog":
            demographics["dogs"] += 1
        elif detection.get("class") == "person":
            if detection.get("age") == "child":
                demographics["children"] += 1
            else:
                demographics["adults"] += 1
            
            if detection.get("gender") == "male":
                demographics["males"] += 1
            elif detection.get("gender") == "female":
                demographics["females"] += 1
    
    return demographics

def format_results(demographics: Dict[str, int], hourly_data: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
    results = {
        "timestamp": get_timestamp(),
        "summary": demographics
    }
    
    if hourly_data:
        results["hourly_breakdown"] = hourly_data
    
    return results
