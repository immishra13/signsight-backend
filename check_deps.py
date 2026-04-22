#!/usr/bin/env python3
"""
Check and install required dependencies
"""

import subprocess
import sys
from pathlib import Path

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except Exception as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def check_dependencies():
    """Check and install all required dependencies"""
    print("🔍 Checking dependencies...")
    
    packages = [
        "websockets",
        "opencv-python",
        "numpy",
        "ultralytics==8.3.50",
        "mediapipe==0.10.14"
    ]
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        success = install_package(package)
        if success:
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
    
    print("\n🧪 Testing imports...")
    
    # Test imports
    try:
        import cv2
        print("✅ OpenCV import successful")
    except:
        print("❌ OpenCV import failed")
    
    try:
        import numpy as np
        print("✅ NumPy import successful")
    except:
        print("❌ NumPy import failed")
    
    try:
        import websockets
        print("✅ WebSockets import successful")
    except:
        print("❌ WebSockets import failed")
    
    try:
        from ultralytics import YOLO
        print("✅ YOLO import successful")
    except Exception as e:
        print(f"❌ YOLO import failed: {e}")
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe import successful")
    except Exception as e:
        print(f"❌ MediaPipe import failed: {e}")
    
    print("\n✅ Dependency check complete!")

if __name__ == "__main__":
    check_dependencies()
