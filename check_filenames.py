"""
UTILITY: Check Filename Parity Between Images and Labels
=========================================================
Verifies that every image in folder A has a matching label in folder B
and vice-versa. Mismatches will cause YOLO to crash mid-training.

Usage:
    python check_filenames.py --images ./dataset/images/train \
                               --labels ./dataset/labels/train
"""

import os
import argparse


def check_parity(images_dir, labels_dir):
    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        return
    if not os.path.isdir(labels_dir):
        print(f"[ERROR] Labels directory not found: {labels_dir}")
        return

    img_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    lbl_stems = {
        os.path.splitext(f)[0]
        for f in os.listdir(labels_dir)
        if f.endswith(".txt")
    }

    missing_labels = sorted(img_stems - lbl_stems)
    missing_images = sorted(lbl_stems - img_stems)
    matched        = img_stems & lbl_stems

    print(f"\n{'='*55}")
    print(f"Parity Check")
    print(f"  Images dir : {images_dir}")
    print(f"  Labels dir : {labels_dir}")
    print(f"{'='*55}")
    print(f"  ✓ Matched pairs      : {len(matched)}")
    print(f"  ✗ Images w/o label   : {len(missing_labels)}")
    print(f"  ✗ Labels w/o image   : {len(missing_images)}")

    if missing_labels:
        print(f"\n[IMAGES WITHOUT LABELS] (first 20):")
        for name in missing_labels[:20]:
            print(f"  {name}.jpg")

    if missing_images:
        print(f"\n[LABELS WITHOUT IMAGES] (first 20):")
        for name in missing_images[:20]:
            print(f"  {name}.txt")

    if not missing_labels and not missing_images:
        print("\n[OK] All files are perfectly paired. Ready to train!")
    else:
        print(f"\n[WARN] Fix the mismatches above before training.")


def main():
    parser = argparse.ArgumentParser(description="Image/Label parity checker.")
    parser.add_argument("--images", required=True, help="Path to images folder")
    parser.add_argument("--labels", required=True, help="Path to labels folder")
    args = parser.parse_args()
    check_parity(args.images, args.labels)


if __name__ == "__main__":
    main()
