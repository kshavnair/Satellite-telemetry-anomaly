"""
Kaggle Setup Script for Telemetry Anomaly Detection
Run this script to pull the Kaggle kernel and dataset.

Prerequisites:
1. Install Kaggle CLI: pip install kaggle
2. Create Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Create API token (downloads kaggle.json)
   - Place kaggle.json in: %USERPROFILE%\\.kaggle\\
"""
import subprocess
import sys
import os

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("ISRO Telemetry Anomaly Detection - Kaggle Setup")
    print("=" * 60)
    
    # Pull the kernel
    print("\n[1/2] Pulling Kaggle kernel: devraai/satellite-telemetry-anomaly-prediction")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "pull", 
             "devraai/satellite-telemetry-anomaly-prediction", "-p", project_dir],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   [OK] Kernel pulled successfully!")
        else:
            print(f"   [ERROR] {result.stderr}")
    except FileNotFoundError:
        print("   [ERROR] Kaggle CLI not found. Run: pip install kaggle")
        return 1
    
    # Pull the satellite telemetry dataset
    print("\n[2/2] Pulling dataset: orvile/satellite-telemetry-data-anomaly-prediction")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "orvile/satellite-telemetry-data-anomaly-prediction", "-p", project_dir],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   [OK] Dataset downloaded successfully!")
        else:
            print(f"   [ERROR] {result.stderr}")
    except FileNotFoundError:
        print("   [ERROR] Kaggle CLI not found.")
        return 1
    
    print("\n" + "=" * 60)
    print("Setup complete! Run: python telemetry_anomaly_model.py")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
