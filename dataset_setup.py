"""
dataset_setup.py
=================
Helper script to download and prepare the Stanford Dogs Dataset
(or any Kaggle dog breed dataset) into the required folder structure.

Folder structure expected by the model:
    dataset/
    └── train/
        ├── beagle/
        │   ├── img001.jpg
        │   └── ...
        ├── golden_retriever/
        └── ...

Run this script ONCE before training.

Author: Nishita Thakur
"""

import os
import urllib.request
import tarfile

# ── Option 1: Download Stanford Dogs Dataset ──────────────────────────────────
# The Stanford Dogs Dataset contains 20,580 images across 120 breeds.
# Source: http://vision.stanford.edu/aditya86/ImageNetDogs/

STANFORD_IMAGES_URL  = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
STANFORD_ANNOT_URL   = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
DOWNLOAD_DIR         = "raw_data"
DATASET_DIR          = "dataset/train"


def download_stanford_dogs():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    images_tar = os.path.join(DOWNLOAD_DIR, "images.tar")

    print("[INFO] Downloading Stanford Dogs Dataset images (~750 MB)...")
    urllib.request.urlretrieve(STANFORD_IMAGES_URL, images_tar)
    print("[INFO] Download complete. Extracting...")

    with tarfile.open(images_tar, "r") as tar:
        tar.extractall(DOWNLOAD_DIR)
    print("[INFO] Extraction complete.")


def organize_dataset():
    """
    Renames and moves images from Stanford's nested folder structure
    into dataset/train/<breed_name>/ folders.
    """
    import shutil
    import re

    images_root = os.path.join(DOWNLOAD_DIR, "Images")
    if not os.path.exists(images_root):
        print("[ERROR] Images folder not found. Run download_stanford_dogs() first.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    breed_folders = sorted(os.listdir(images_root))

    print(f"[INFO] Organizing {len(breed_folders)} breeds...")
    for folder in breed_folders:
        # Stanford folder names look like: n02085620-Chihuahua
        match = re.search(r"-(.*)", folder)
        if not match:
            continue
        breed_name = match.group(1).lower().replace(" ", "_")
        src = os.path.join(images_root, folder)
        dst = os.path.join(DATASET_DIR, breed_name)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"  ✔ {breed_name} ({len(os.listdir(dst))} images)")

    print(f"\n[INFO] Dataset ready at: {DATASET_DIR}/")
    print(f"[INFO] Total breeds: {len(os.listdir(DATASET_DIR))}")


# ── Option 2: Use your own dataset ────────────────────────────────────────────
# Simply place your images manually:
#
#   dataset/train/
#       labrador/   → add labrador images here
#       poodle/     → add poodle images here
#       ...
#
# Minimum recommended: 100+ images per breed for good accuracy.


if __name__ == "__main__":
    print("=" * 50)
    print(" Dog Breed Dataset Setup")
    print("=" * 50)
    print("\nOptions:")
    print("  1. Download Stanford Dogs Dataset (120 breeds, ~750 MB)")
    print("  2. Use my own dataset (manual folder setup)")
    choice = input("\nEnter choice (1/2): ").strip()

    if choice == "1":
        download_stanford_dogs()
        organize_dataset()
        print("\n[DONE] Dataset is ready. Run training with:")
        print("  python dog_breed_prediction.py --mode train")
    elif choice == "2":
        print("\n[INFO] Please organize your images like this:")
        print("""
    dataset/
    └── train/
        ├── breed_one/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── breed_two/
            ├── img1.jpg
            └── img2.jpg
        """)
        print("Then run: python dog_breed_prediction.py --mode train")
    else:
        print("[ERROR] Invalid choice.")
