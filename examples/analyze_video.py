import argparse
from crowdinsight import CrowdAnalyzer
from tqdm import tqdm
import os

def main():
    """Analyze a video file and print detailed analytics."""
    parser = argparse.ArgumentParser(description="CrowdInsight Video Analysis")
    parser.add_argument("--video", type=str, default="videos/cctv.mp4", help="Path to video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--output", type=str, default="outputs/analysis_results.json", help="Output path for results")
    parser.add_argument("--no-show", action="store_true", help="Disable video display")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return

    print("\n=== CrowdInsight Video Analysis ===")
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print("====================================\n")

    try:
        analyzer = CrowdAnalyzer(
            video_source=args.video,
            model_path=args.model,
            conf_threshold=args.conf,
            show_video=not args.no_show
        )

        # Run analysis and save results
        results = analyzer.run_analysis(output_path=args.output)

        # Print summary
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

        # Print dwell times for tracked objects
        print("\nDwell Times (seconds):")
        for track_id, dwell_time in results.get('dwell_times', {}).items():
            print(f"Track {track_id}: {dwell_time:.2f} seconds")

        print(f"\nResults saved to: {args.output}")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main()
