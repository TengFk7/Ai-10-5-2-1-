# -*- coding: utf-8 -*-
"""
Download & Merge Thai Coin Datasets from Roboflow
===================================================
Downloads two datasets and merges them into one unified dataset.

Dataset 1: coin-thai v9 (data-uo5dw)
  Classes: 01(=1 Baht), 10(=10 Baht), 5(=5 Baht)

Dataset 2: thai-coins-model v4 (kittikun)
  Classes: 1 bahts, 2 bahts, 5 bahts, 10 bahts

Merged (unified):
  0: 1-baht
  1: 2-baht
  2: 5-baht
  3: 10-baht

Usage:
  python download_dataset.py
"""

import os
import sys
import shutil
import glob
import yaml

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def download_dataset_1():
    """Download coin-thai v9 from data-uo5dw."""
    from roboflow import Roboflow

    print("[Dataset 1] coin-thai v9 (data-uo5dw)")
    print("  Connecting...")
    rf = Roboflow(api_key="dsVxDrZmAROD9NSeYsti")
    project = rf.workspace("data-uo5dw").project("coin-thai")
    version = project.version(9)

    print("  Downloading...")
    dataset = version.download("yolov8")
    print(f"  Downloaded to: {dataset.location}")
    return dataset.location


def download_dataset_2():
    """Download thai-coins-model v4 from kittikun."""
    from roboflow import Roboflow

    print("[Dataset 2] thai-coins-model v4 (kittikun)")
    print("  Connecting...")
    rf = Roboflow(api_key="dsVxDrZmAROD9NSeYsti")
    project = rf.workspace("kittikun").project("thai-coins-model")
    version = project.version(4)

    print("  Downloading...")
    dataset = version.download("yolov8")
    print(f"  Downloaded to: {dataset.location}")
    return dataset.location


def read_class_names(yaml_path):
    """Read class names from data.yaml."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    names = config.get('names', [])
    if isinstance(names, dict):
        # Convert {0: 'name', 1: 'name'} to list
        names = [names[k] for k in sorted(names.keys())]
    return names


def build_class_mapping(source_names, unified_names):
    """
    Build mapping from source class IDs to unified class IDs.

    Handles various naming conventions:
      '01', '1 bahts', '1-baht', '1baht' -> 1-baht (unified 0)
      '2 bahts', '2-baht'                -> 2-baht (unified 1)
      '5', '5 bahts', '5-baht'           -> 5-baht (unified 2)
      '10', '10 bahts', '10-baht'        -> 10-baht (unified 3)
    """
    # Normalize a class name to just the number
    def normalize(name):
        name = name.lower().strip()
        # Remove common suffixes
        for suffix in [' bahts', ' baht', '-baht', '-bahts', 'bahts', 'baht']:
            name = name.replace(suffix, '')
        name = name.strip()
        # Handle '01' -> '1'
        if name == '01':
            name = '1'
        return name

    # Build value -> unified_id mapping
    unified_map = {}
    for uid, uname in enumerate(unified_names):
        val = normalize(uname)
        unified_map[val] = uid

    # Build source_id -> unified_id mapping
    mapping = {}
    for sid, sname in enumerate(source_names):
        val = normalize(sname)
        if val in unified_map:
            mapping[sid] = unified_map[val]
        else:
            print(f"  WARNING: Cannot map class '{sname}' (normalized: '{val}') - skipping")
            mapping[sid] = None

    return mapping


def remap_label_file(src_path, dst_path, class_mapping):
    """
    Read a YOLO label file, remap class IDs, and write to destination.
    Returns number of valid annotations.
    """
    valid_lines = []
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            old_class = int(parts[0])
            new_class = class_mapping.get(old_class)
            if new_class is None:
                continue  # Skip unmapped classes
            parts[0] = str(new_class)
            valid_lines.append(' '.join(parts))

    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_lines))
        if valid_lines:
            f.write('\n')

    return len(valid_lines)


def copy_split(dataset_dir, split_name, merged_dir, class_mapping, prefix, img_count_start=0):
    """
    Copy images and remapped labels from one dataset split to merged directory.

    split_name: 'train', 'valid', 'test'
    prefix: unique prefix to avoid filename collisions (e.g. 'ds1_', 'ds2_')

    Returns: number of images copied
    """
    # Roboflow uses 'train', 'valid', 'test' folders
    src_images = os.path.join(dataset_dir, split_name, 'images')
    src_labels = os.path.join(dataset_dir, split_name, 'labels')

    if not os.path.exists(src_images):
        return 0

    # Map 'valid' -> 'val' for our merged dataset naming
    dst_split = 'val' if split_name == 'valid' else split_name
    dst_images = os.path.join(merged_dir, 'images', dst_split)
    dst_labels = os.path.join(merged_dir, 'labels', dst_split)
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    copied = 0

    for fname in os.listdir(src_images):
        if not fname.lower().endswith(img_exts):
            continue

        # Add prefix to avoid collisions between datasets
        new_fname = f"{prefix}{fname}"
        base_name = os.path.splitext(fname)[0]
        new_base = os.path.splitext(new_fname)[0]

        # Copy image
        shutil.copy2(
            os.path.join(src_images, fname),
            os.path.join(dst_images, new_fname)
        )

        # Copy and remap label
        label_file = os.path.join(src_labels, base_name + '.txt')
        if os.path.exists(label_file):
            remap_label_file(
                label_file,
                os.path.join(dst_labels, new_base + '.txt'),
                class_mapping
            )
        else:
            # Create empty label file (negative example)
            open(os.path.join(dst_labels, new_base + '.txt'), 'w').close()

        copied += 1

    return copied


def merge_datasets(ds1_dir, ds2_dir, merged_dir):
    """
    Merge two Roboflow YOLOv8 datasets into one unified dataset.
    Remaps all class IDs to the unified scheme.
    """
    print()
    print("=" * 60)
    print("  Merging datasets...")
    print("=" * 60)

    # Unified class names (4 classes)
    unified_names = ['1-baht', '2-baht', '5-baht', '10-baht']

    # Read class names from each dataset
    ds1_yaml = os.path.join(ds1_dir, 'data.yaml')
    ds2_yaml = os.path.join(ds2_dir, 'data.yaml')

    ds1_classes = read_class_names(ds1_yaml)
    ds2_classes = read_class_names(ds2_yaml)

    print(f"\n  Dataset 1 classes: {ds1_classes}")
    print(f"  Dataset 2 classes: {ds2_classes}")
    print(f"  Unified classes:   {unified_names}")

    # Build class mappings
    ds1_mapping = build_class_mapping(ds1_classes, unified_names)
    ds2_mapping = build_class_mapping(ds2_classes, unified_names)

    print(f"\n  Dataset 1 mapping: {ds1_mapping}")
    print(f"  Dataset 2 mapping: {ds2_mapping}")

    # Clean merged directory
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)

    # Copy each split
    print()
    total_stats = {}
    for split in ['train', 'valid', 'test']:
        n1 = copy_split(ds1_dir, split, merged_dir, ds1_mapping, 'ct_')
        n2 = copy_split(ds2_dir, split, merged_dir, ds2_mapping, 'tc_')
        dst_split = 'val' if split == 'valid' else split
        total_stats[dst_split] = n1 + n2
        print(f"  {split:>5s}: {n1} (coin-thai) + {n2} (thai-coins-model) = {n1 + n2} images")

    # Write unified data.yaml
    yaml_content = {
        'path': os.path.abspath(merged_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 4,
        'names': unified_names,
    }

    yaml_path = os.path.join(merged_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"\n  Merged data.yaml: {yaml_path}")
    print(f"\n  Total: {sum(total_stats.values())} images")
    for split, count in total_stats.items():
        print(f"    {split}: {count}")

    return merged_dir, yaml_path


def find_dataset_dir(patterns):
    """Find an existing dataset directory matching any of the given patterns."""
    for pattern in patterns:
        if os.path.exists(pattern) and os.path.exists(os.path.join(pattern, 'data.yaml')):
            return pattern
    # Also try glob for case-insensitive search
    for pattern in glob.glob("*"):
        if os.path.isdir(pattern) and os.path.exists(os.path.join(pattern, 'data.yaml')):
            name_lower = pattern.lower()
            for p in patterns:
                if os.path.basename(p).lower() == name_lower:
                    return pattern
    return None


def main():
    print()
    print("=" * 60)
    print("  Download & Merge Thai Coin Datasets")
    print("=" * 60)
    print()

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow not installed!")
        print("  Run: pip install roboflow")
        return

    merged_dir = os.path.join(os.getcwd(), 'merged-dataset')

    # --- Dataset 1 ---
    ds1_dir = find_dataset_dir(['coin-thai-9', 'Coin-thai-9'])
    if ds1_dir:
        print(f"[Dataset 1] Already exists: {ds1_dir}")
    else:
        ds1_dir = download_dataset_1()

    print()

    # --- Dataset 2 ---
    ds2_dir = find_dataset_dir([
        'thai-coins-model-4', 'Thai-Coins-Model-4',
        'Thai-coins-model-4', 'thai-Coins-Model-4'
    ])
    if ds2_dir:
        print(f"[Dataset 2] Already exists: {ds2_dir}")
    else:
        ds2_dir = download_dataset_2()

    # Verify both datasets exist
    if not os.path.exists(os.path.join(ds1_dir, 'data.yaml')):
        print(f"ERROR: Dataset 1 data.yaml not found at: {ds1_dir}")
        return
    if not os.path.exists(os.path.join(ds2_dir, 'data.yaml')):
        print(f"ERROR: Dataset 2 data.yaml not found at: {ds2_dir}")
        return

    # Merge
    merged_dir, yaml_path = merge_datasets(ds1_dir, ds2_dir, merged_dir)

    print()
    print("=" * 60)
    print("  Download & Merge Complete!")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("    python train.py        -> Train the model")
    print("    python app.py          -> Run detection app")
    print("    python app.py --demo   -> Run in demo mode")
    print("=" * 60)


if __name__ == "__main__":
    main()
