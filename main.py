"""
Financial Document AI - Main Entry Point

End-to-end pipeline for financial table understanding.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.utils import load_config, get_device


def main():
    parser = argparse.ArgumentParser(
        description="Financial Document AI - Table Understanding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline experiments
  python main.py --mode baseline --num_samples 100
  
  # Process a PDF file (future)
  python main.py --mode process --input report.pdf
        """
    )
    
    parser.add_argument("--mode", type=str, choices=["baseline", "process", "demo"],
                       default="baseline", help="Operation mode")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples for baseline experiments")
    parser.add_argument("--input", type=str, help="Input file for processing")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Financial Document AI")
    print("Table Understanding Pipeline")
    print("=" * 60)
    
    # Check device
    device = get_device()
    
    if args.mode == "baseline":
        print("\nRunning baseline experiments...")
        from experiments.run_all_baselines import run_all_baselines
        run_all_baselines(args.config, args.num_samples)
        
    elif args.mode == "process":
        if not args.input:
            print("Error: --input required for process mode")
            sys.exit(1)
        print(f"\nProcessing: {args.input}")
        print("(PDF processing not yet implemented)")
        # TODO: Implement PDF processing pipeline
        
    elif args.mode == "demo":
        print("\nRunning demo...")
        print("(Demo not yet implemented)")
        # TODO: Implement demo mode


if __name__ == "__main__":
    main()asd
