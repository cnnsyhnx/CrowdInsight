from crowdinsight import CrowdAnalyzer
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CrowdInsight Live Stream Analysis")
    parser.add_argument("--source", type=int, default=0, help="Camera source (default: 0 for webcam)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output", type=str, help="Output path for results (optional)")
    args = parser.parse_args()

    # Initialize analyzer with live stream
    analyzer = CrowdAnalyzer(
        video_source=args.source,
        model_path=args.model,
        conf_threshold=args.conf,
        show_video=True
    )
    
    try:
        # Run live analysis
        results = analyzer.run_live_stream()
        
        # Save results if output path is provided
        if args.output:
            analyzer.export_results(args.output)
            print(f"\nResults saved to: {args.output}")
        
        # Print final summary
        print("\nAnalysis Summary:")
        print(f"Total Visitors: {results['summary']['total_visitors']}")
        print(f"Adults: {results['summary']['adults']}")
        print(f"Children: {results['summary']['children']}")
        print(f"Males: {results['summary']['males']}")
        print(f"Females: {results['summary']['females']}")
        print(f"Dogs: {results['summary']['dogs']}")
        
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
