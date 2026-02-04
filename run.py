#!/usr/bin/env python3
"""
AI Knowledge Assistant - Application Runner

Simple script to run the Streamlit application with proper configuration.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the Streamlit application."""
    # Get the path to the main app
    app_path = Path(__file__).parent / "app" / "main.py"
    
    if not app_path.exists():
        print(f"Error: Could not find {app_path}")
        sys.exit(1)
    
    # Run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("🚀 Starting AI Knowledge Assistant...")
    print(f"📍 Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
