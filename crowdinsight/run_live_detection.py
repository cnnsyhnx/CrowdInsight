import argparse
from crowdinsight.detector import ObjectDetector
import sys
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='CrowdInsight Live Detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory for JSON files (default: outputs)')
    parser.add_argument('--window-name', type=str, default='CrowdInsight Live Detection', help='Window name for the display (default: CrowdInsight Live Detection)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)
    print("\n=== CrowdInsight Live Detection ===")
    print(f"Camera ID: {args.camera}")
    print(f"Output Directory: {args.output}")
    print("Press 'q' to quit")
    print("================================\n")
    try:
        detector = ObjectDetector()
        detector.run_live_detection(
            camera_id=args.camera,
            window_name=args.window_name
        )
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 