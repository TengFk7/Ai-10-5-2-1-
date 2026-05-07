# -*- coding: utf-8 -*-
"""
Thai Coin Detection - YOLOv8 Training Script
=============================================
Merged datasets:
  - coin-thai v9 (data-uo5dw)
  - thai-coins-model v4 (kittikun)

Unified Classes (4):
  0: 1-baht   (1 Baht)
  1: 2-baht   (2 Baht)
  2: 5-baht   (5 Baht)
  3: 10-baht  (10 Baht)

Usage:
  1. python download_dataset.py   (download & merge datasets)
  2. python train.py              (train model)
  3. python train.py val          (validate model)
"""

from ultralytics import YOLO
import os
import sys
import glob
import yaml


def find_dataset_yaml():
    """
    Find the Roboflow dataset's data.yaml file.
    Searches common download locations.
    """
    # Priority order of paths to search (merged dataset first)
    search_paths = [
        # Merged dataset (preferred - combines both Roboflow datasets)
        os.path.join("merged-dataset", "data.yaml"),
        # Individual Roboflow downloads
        os.path.join("coin-thai-9", "data.yaml"),
        os.path.join("thai-coins-model-4", "data.yaml"),
        os.path.join("Thai-coins-model-4", "data.yaml"),
        # Our data directory
        os.path.join("data", "data.yaml"),
        os.path.join("data", "dataset.yaml"),
    ]
    
    # Also search for any data.yaml in subdirectories (1 level deep)
    for pattern in glob.glob("*/data.yaml"):
        if pattern not in search_paths:
            search_paths.append(pattern)
    
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None


def check_dataset(yaml_path):
    """Verify dataset structure and report stats."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dataset_root = os.path.dirname(yaml_path)
    
    # Roboflow uses 'train', 'valid', 'test' folders
    train_dir = config.get('train', 'train/images')
    val_dir = config.get('val', 'valid/images')
    
    # Handle relative paths
    if not os.path.isabs(train_dir):
        train_path = os.path.join(dataset_root, train_dir)
    else:
        train_path = train_dir
        
    if not os.path.isabs(val_dir):
        val_path = os.path.join(dataset_root, val_dir)
    else:
        val_path = val_dir
    
    print(f"📄 Dataset YAML: {yaml_path}")
    print(f"📁 Dataset root: {dataset_root}")
    print()
    
    # Show classes
    names = config.get('names', {})
    nc = config.get('nc', len(names))
    print(f"🏷️  Classes ({nc}):")
    if isinstance(names, list):
        for i, name in enumerate(names):
            print(f"   {i}: {name}")
    elif isinstance(names, dict):
        for k, v in names.items():
            print(f"   {k}: {v}")
    print()
    
    # Count images
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    if os.path.exists(train_path):
        train_count = len([f for f in os.listdir(train_path) if f.lower().endswith(img_exts)])
        print(f"📸 Train images: {train_count}")
    else:
        train_count = 0
        print(f"⚠️  Train path not found: {train_path}")
    
    if os.path.exists(val_path):
        val_count = len([f for f in os.listdir(val_path) if f.lower().endswith(img_exts)])
        print(f"📸 Valid images: {val_count}")
    else:
        val_count = 0
        print(f"⚠️  Valid path not found: {val_path}")
    
    print()
    return train_count > 0 and val_count > 0


def train():
    """Train YOLOv8 model using the Roboflow dataset."""
    
    print("=" * 60)
    print("  🚀 Thai Coin Detection - YOLOv8 Training")
    print("  Dataset: Roboflow coin-thai v9")
    print("=" * 60)
    print()
    
    # Find dataset YAML
    yaml_path = find_dataset_yaml()
    
    if yaml_path is None:
        print("❌ Dataset not found!")
        print()
        print("📋 กรุณาดาวน์โหลด dataset ก่อน:")
        print("   python download_dataset.py")
        print()
        print("หรือดาวน์โหลดเองด้วย:")
        print("   pip install roboflow")
        print("   python -c \"")
        print("   from roboflow import Roboflow")
        print("   rf = Roboflow(api_key='dsVxDrZmAROD9NSeYsti')")
        print("   project = rf.workspace('data-uo5dw').project('coin-thai')")
        print("   version = project.version(9)")
        print("   dataset = version.download('yolov8')")
        print("   \"")
        print("=" * 60)
        return
    
    # Verify dataset
    if not check_dataset(yaml_path):
        print("❌ Dataset verification failed!")
        print("   กรุณาตรวจสอบว่า dataset ถูกดาวน์โหลดครบ")
        return
    
    print("🔧 Training Configuration:")
    print(f"   Base model: yolov8n.pt (nano - fast)")
    print(f"   Dataset:    {yaml_path}")
    print(f"   Epochs:     100")
    print(f"   Image size: 640")
    print(f"   Batch:      16")
    print()
    
    # Load pretrained YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    
    # Train with the Roboflow dataset
    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        name='thai_coins',
        patience=20,           # early stopping
        save=True,
        save_period=10,
        device='0',            # GPU (use 'cpu' if no GPU)
        workers=4,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        augment=True,
        # Data augmentation (เหรียญหมุนได้ทุกมุม)
        hsv_h=0.015,          # hue augmentation
        hsv_s=0.7,            # saturation augmentation
        hsv_v=0.4,            # value augmentation
        degrees=180,          # rotation (coins can be any angle)
        translate=0.1,
        scale=0.5,
        flipud=0.5,           # flip up-down
        fliplr=0.5,           # flip left-right
        mosaic=1.0,           # mosaic augmentation
    )
    
    print()
    print("=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"📁 Best model: runs/detect/thai_coins/weights/best.pt")
    print(f"📁 Last model: runs/detect/thai_coins/weights/last.pt")
    print()
    
    # Auto-copy best model to models directory
    best_model = os.path.join('runs', 'detect', 'thai_coins', 'weights', 'best.pt')
    if os.path.exists(best_model):
        os.makedirs('models', exist_ok=True)
        import shutil
        dest = os.path.join('models', 'thai_coins_best.pt')
        shutil.copy2(best_model, dest)
        print(f"✅ Best model copied to: {dest}")
        print()
        print("📋 Next steps:")
        print("   python app.py   → Start the detection app!")
    print("=" * 60)


def validate():
    """Validate the trained model."""
    model_path = os.path.join('models', 'thai_coins_best.pt')
    if not os.path.exists(model_path):
        print("❌ Model not found! Train first: python train.py")
        return
    
    yaml_path = find_dataset_yaml()
    if yaml_path is None:
        print("❌ Dataset YAML not found!")
        return
    
    print("🔍 Validating model...")
    model = YOLO(model_path)
    results = model.val(data=yaml_path)
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'val':
        validate()
    else:
        train()
