from crowdinsight import CrowdAnalyzer

def main():
    # Initialize analyzer with video file
    analyzer = CrowdAnalyzer(
        video_source="videos/cctv.mp4",
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        show_video=True
    )
    
    # Run analysis and save results
    results = analyzer.run_analysis(output_path="outputs/analysis_results.json")
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total Visitors: {results['summary']['total_visitors']}")
    print(f"Adults: {results['summary']['adults']}")
    print(f"Children: {results['summary']['children']}")
    print(f"Males: {results['summary']['males']}")
    print(f"Females: {results['summary']['females']}")
    print(f"Dogs: {results['summary']['dogs']}")
    
    print("\nHourly Breakdown:")
    for hour, data in results['hourly_breakdown'].items():
        print(f"{hour}: {data['visitors']} visitors")

if __name__ == "__main__":
    main()
