"""
Utility script to run all preprocessing steps.
"""

import subprocess
import sys


def run_script(script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*70}")
    print(f"Running {script_name}")
    print(f"{'='*70}\n")
    
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_name}")
        sys.exit(1)
    
    print(f"\n✅ {script_name} completed successfully")


def main():
    print("\n" + "="*70)
    print("MUSEAI - FULL PREPROCESSING PIPELINE")
    print("="*70)
    print("\nThis will process all style and content images.")
    print("Make sure you have placed:")
    print("  - Picasso paintings in data/style_raw/picasso/")
    print("  - Rembrandt paintings in data/style_raw/rembrandt/")
    print("  - Face images in data/content/faces/raw/")
    print("\n" + "="*70)
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Preprocessing cancelled.")
        return
    
    # Run preprocessing scripts
    run_script('src/preprocess/prepare_style_images.py')
    run_script('src/preprocess/prepare_content_images.py')
    
    print("\n" + "="*70)
    print("✅ ALL PREPROCESSING COMPLETE!")
    print("="*70)

if __name__ == '__main__':
    main()
