#!/usr/bin/env python3
"""
Download Roboflow Dental Caries Dataset

Requires a valid Roboflow API key (roboflow.com -> Account -> Roboflow Keys):
  export ROBOFLOW_API_KEY=your_key_here
"""

import os
import sys

# Install roboflow if needed
os.system("pip install -q roboflow")

from roboflow import Roboflow

api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    sys.exit("ROBOFLOW_API_KEY not set. Get a key from roboflow.com -> Account -> Roboflow Keys.")

print("Downloading Roboflow Dental Caries Dataset...")
print("=" * 60)

# Initialize Roboflow with API key
rf = Roboflow(api_key=api_key)

# Get project and download dataset
project = rf.workspace("project-group13").project("dental-caries-detection-using-dl")
dataset = project.version(10).download("yolov8")

print("=" * 60)
print(f"Dataset downloaded to: {dataset.location}")
print("Ready for training with YOLOv8!")
