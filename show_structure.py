#!/usr/bin/env python3

import os
from pathlib import Path

def generate_file_tree(root_dir, prefix="", ignore_dirs={'.git', '__pycache__', 'node_modules', 'venv', '.venv'}):
    root_path = Path(root_dir)
    files = []
    
    items = sorted(root_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    for i, item in enumerate(items):
        if item.name.startswith('.') and item.name not in {'.env.example', '.gitignore'}:
            continue
        
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        if item.is_dir():
            if item.name in ignore_dirs:
                continue
            files.append(prefix + current_prefix + item.name + "/")
            files.extend(generate_file_tree(item, prefix + next_prefix, ignore_dirs))
        else:
            files.append(prefix + current_prefix + item.name)
    
    return files

def main():
    print("PROJECT FILE STRUCTURE")
    print("=" * 70)
    print()
    
    tree = generate_file_tree(".")
    for line in tree:
        print(line)
    
    print()
    print("=" * 70)
    print("TOTAL FILES AND DIRECTORIES")
    print("=" * 70)
    
    file_count = sum(1 for line in tree if not line.rstrip().endswith("/"))
    dir_count = sum(1 for line in tree if line.rstrip().endswith("/"))
    
    print(f"Directories: {dir_count}")
    print(f"Files: {file_count}")
    print(f"Total: {file_count + dir_count}")
    
    print()
    print("=" * 70)
    print("KEY FILES BY CATEGORY")
    print("=" * 70)
    
    categories = {
        "Backend": [
            "backend/app.py",
            "backend/config.py",
            "backend/models/classifier.py",
            "backend/utils/pose_extractor.py",
            "backend/utils/shot_analyzer.py"
        ],
        "Training": [
            "training/dataset.py",
            "training/train.py",
            "training/export.py",
            "training/run_training.py"
        ],
        "Frontend": [
            "frontend/src/App.js",
            "frontend/src/components/ImagePredictor.js",
            "frontend/src/components/VideoPredictor.js",
            "frontend/src/components/LiveAnalytics.js",
            "frontend/src/components/ModelInfo.js"
        ],
        "Utilities": [
            "setup.py",
            "extract_poses.py",
            "predict.py",
            "annotate_video.py",
            "validate_data.py",
            "performance_monitor.py",
            "test_api.py"
        ],
        "Configuration": [
            "requirements.txt",
            "docker-compose.yml",
            "Dockerfile.backend",
            "frontend/Dockerfile",
            ".env.example",
            ".gitignore"
        ],
        "Documentation": [
            "SETUP_GUIDE.txt",
            "PROJECT_REFERENCE.txt"
        ]
    }
    
    for category, files in categories.items():
        print(f"\n{category}:")
        for f in files:
            exists = Path(f).exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {f}")

if __name__ == "__main__":
    main()
