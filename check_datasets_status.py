#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

print("=" * 70)
print("DENTAL DATASETS - COMPREHENSIVE STATUS REPORT")
print("=" * 70)
print()

datasets_dir = Path(r"d:\git\dental\medclip\datasets")

if not datasets_dir.exists():
    print("❌ Datasets directory not found!")
    exit(1)

# Analyze downloaded files
print("📦 DOWNLOADED FILES:")
print("-" * 70)

total_size = 0
files_info = []

for fpath in datasets_dir.glob("*"):
    if fpath.is_file():
        size_mb = fpath.stat().st_size / (1024 * 1024)
        total_size += fpath.stat().st_size
        files_info.append((fpath.name, size_mb))
        print(f"  ✓ {fpath.name:<40} {size_mb:>10.1f} MB")

print()
print(f"Total downloaded: {total_size / (1024**3):.2f} GB ({total_size / (1024**2):.0f} MB)")
print()

# Analyze extracted directories
print("📂 EXTRACTED DIRECTORIES:")
print("-" * 70)

extracted_count = 0
for dpath in datasets_dir.glob("*"):
    if dpath.is_dir():
        file_count = sum(1 for _ in dpath.rglob("*") if _.is_file())
        size = sum(f.stat().st_size for f in dpath.rglob("*") if f.is_file())
        extracted_count += 1
        print(f"  ✓ {dpath.name:<40} {file_count:>6} files ({size / (1024**2):>8.1f} MB)")

print()

# Dataset completeness
print("📋 DATASET STATUS:")
print("-" * 70)

datasets_expected = {
    "DENTEX": ["training_data.zip", "test_data.zip", "validation_data.zip"],
    "Zenodo Caries": ["Zenodo_Dental_Caries.zip"],
    "Roboflow Caries": ["Roboflow_Caries"],
    "Mandible Segmentation": ["Mandible_Segmentation.zip"],
    "Tufts Dental": ["Requires manual download from tdd.ece.tufts.edu"]
}

for dataset, items in datasets_expected.items():
    found = False
    for item in items:
        if (datasets_dir / item).exists():
            found = True
            print(f"  ✅ {dataset:<35} READY")
            break
    if not found:
        print(f"  ❌ {dataset:<35} NOT FOUND")

print()
print("=" * 70)
print("SUMMARY:")
print("-" * 70)
print(f"  Total files downloaded: {len(files_info)}")
print(f"  Total directories extracted: {extracted_count}")
print(f"  Total storage used: {total_size / (1024**3):.2f} GB")
print("=" * 70)
