#!/usr/bin/env python3
"""
Improved Dental Datasets Downloader with Error Handling
"""

import os
import sys
import urllib.request
import shutil
from pathlib import Path
from zipfile import ZipFile

print("=" * 70)
print("DENTAL DATASETS DOWNLOADER - IMPROVED VERSION")
print("=" * 70)
print()

# Setup
datasets_dir = Path(r"d:\git\dental\medclip\datasets")
datasets_dir.mkdir(exist_ok=True)

# Define datasets with verified URLs
datasets = [
    {
        "name": "DENTEX Training Data (Zenodo)",
        "url": "https://zenodo.org/record/7812323/files/training_data.zip",
        "filename": "DENTEX_training_data_v2.zip",
        "size_expected": "~2.0 GB"
    },
    {
        "name": "DENTEX Test Data (Zenodo)",
        "url": "https://zenodo.org/record/7812323/files/test_data.zip",
        "filename": "DENTEX_test_data.zip",
        "size_expected": "~0.5 GB"
    },
    {
        "name": "Dental Caries (Zenodo)",
        "url": "https://zenodo.org/record/4907880/files/dataset.zip",
        "filename": "Zenodo_Dental_Caries.zip",
        "size_expected": "~300 MB"
    },
    {
        "name": "Mandible Segmentation (Mendeley)",
        "url": "https://data.mendeley.com/public-files/datasets/hxt48yk462/files/c6c81c94-5d61-4e6d-8ec1-fab6f4b90fa3/file_downloaded",
        "filename": "Mandible_Segmentation.zip",
        "size_expected": "~100 MB"
    }
]

print("📥 STARTING DOWNLOADS\n")

completed = 0
failed = 0
skipped = 0

for dataset in datasets:
    filepath = datasets_dir / dataset["filename"]
    
    # Skip if already exists
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"⊘ SKIP: {dataset['name']}")
        print(f"  Already exists ({size_mb:.1f} MB)\n")
        skipped += 1
        continue
    
    print(f"⬇ DOWNLOADING: {dataset['name']}")
    print(f"  Expected: {dataset['size_expected']}")
    print(f"  URL: {dataset['url'][:60]}...")
    
    try:
        urllib.request.urlretrieve(
            dataset["url"],
            filepath,
            reporthook=lambda a, b, c: print(f"  Progress: {(100*a*b//c):3d}%", end="\r") if c > 0 else None
        )
        
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"\n  ✓ Downloaded: {size_mb:.1f} MB\n")
        completed += 1
        
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)[:50]}...\n")
        if filepath.exists():
            filepath.unlink()
        failed += 1

print()
print("=" * 70)
print("EXTRACTION PHASE")
print("=" * 70)
print()

for zipfile_path in datasets_dir.glob("*.zip"):
    extract_dir = datasets_dir / zipfile_path.stem
    
    if extract_dir.exists() and list(extract_dir.iterdir()):
        print(f"⊘ SKIP: {zipfile_path.name}")
        print(f"  Already extracted\n")
        continue
    
    print(f"📂 EXTRACTING: {zipfile_path.name}")
    
    try:
        extract_dir.mkdir(exist_ok=True)
        with ZipFile(zipfile_path, 'r') as z:
            z.extractall(extract_dir)
        
        file_count = sum(1 for _ in extract_dir.rglob("*") if _.is_file())
        print(f"  ✓ Done ({file_count} files)\n")
        
    except Exception as e:
        print(f"  ✗ FAILED: {str(e)[:50]}...\n")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Downloaded: {completed}")
print(f"Failed: {failed}")
print(f"Skipped: {skipped}")
print()
print(f"Location: {datasets_dir}")
print("=" * 70)
