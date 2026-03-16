import os
import glob
import shutil
import random
from PIL import Image

OID_ROOT   = r"C:\Users\lenovo\Desktop\OD\OID\OID_raw"  
DATASET_DIR = "./dataset"              # Output YOLO dataset folder
VAL_RATIO   = 0.20                    
SEED        = 42

# Map OID class folder name → YOLO integer ID
# Key must match the folder name inside OID_ROOT/train/ exactly
CLASS_MAP = {
    "Bottle":   0,
    "Tin can":  1,   # OIDv4 may name it "Tin can" (space) or "Tin_can"
    "Tin_can":  1,   # handle both variants
    "Dice":     2,
    "Ball":     3,
}

CLASS_NAMES = ["Bottle", "Tin can", "Dice", "Ball"]


def make_dirs():
    # Remove old dataset first to avoid mixing old/new classes
    if os.path.isdir(DATASET_DIR):
        print(f"[CLEAN] Removing old dataset at: {DATASET_DIR}/")
        shutil.rmtree(DATASET_DIR)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)
    print(f"[OK] Clean dataset directories created under: {DATASET_DIR}/")


def get_image_size(img_path):
    with Image.open(img_path) as img:
        return img.width, img.height


def convert_oid_label(label_path, img_w, img_h, class_id):
    """
    Parse one OID .txt label file and return YOLO-formatted lines.
    OID format (per line): ClassName left top right bottom
    YOLO format: class_id x_center y_center width height  (normalized)
    """
    yolo_lines = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # parts[0] = class name, parts[1..4] = left top right bottom
            try:
                # OID format: ClassName left top right bottom
                # Class name may have spaces (e.g. "Tin can")
                # So read last 4 tokens as coordinates
                left   = float(parts[-4])
                top    = float(parts[-3])
                right  = float(parts[-2])
                bottom = float(parts[-1])
            except ValueError:
                continue

            x_center = ((left + right)  / 2.0) / img_w
            y_center = ((top  + bottom) / 2.0) / img_h
            width    = (right - left)           / img_w
            height   = (bottom - top)           / img_h

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width    = max(0.0, min(1.0, width))
            height   = max(0.0, min(1.0, height))

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
    return yolo_lines


def collect_samples(oid_split_dir):
    """
    Walk through all class folders in an OID split directory.
    Returns list of (image_path, label_path, class_id) tuples.
    """
    samples = []
    if not os.path.isdir(oid_split_dir):
        print(f"[SKIP] Directory not found: {oid_split_dir}")
        return samples

    for cls_folder in os.listdir(oid_split_dir):
        cls_path = os.path.join(oid_split_dir, cls_folder)
        if not os.path.isdir(cls_path):
            continue

        class_id = CLASS_MAP.get(cls_folder)
        if class_id is None:
            print(f"[WARN] Unknown class folder '{cls_folder}' — skipping.")
            continue

        cls_path = os.path.normpath(cls_path)
        labels_dir = os.path.normpath(os.path.join(cls_path, "Label"))
        images = glob.glob(os.path.join(cls_path, "*.jpg"))

        for img_path in images:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(labels_dir, stem + ".txt")
            if os.path.isfile(lbl_path):
                samples.append((img_path, lbl_path, class_id))
            else:
                print(f"[MISS] No label for image: {img_path}")

    print(f"  Found {len(samples)} labelled samples in: {oid_split_dir}")
    return samples


def write_sample(img_src, lbl_src, class_id, dest_split):
    """Convert + copy one sample into YOLO dataset folder."""
    stem = os.path.splitext(os.path.basename(img_src))[0]
    # Unique stem: prepend class name to avoid collisions across classes
    cls_name = CLASS_NAMES[class_id].replace(" ", "_")
    unique_stem = f"{cls_name}_{stem}"

    img_dst = os.path.join(DATASET_DIR, "images", dest_split, unique_stem + ".jpg")
    lbl_dst = os.path.join(DATASET_DIR, "labels", dest_split, unique_stem + ".txt")

    # Copy image
    shutil.copy2(img_src, img_dst)

    # Convert & write label
    try:
        img_w, img_h = get_image_size(img_src)
    except Exception as e:
        print(f"[WARN] Could not read image size for {img_src}: {e}")
        return False

    yolo_lines = convert_oid_label(lbl_src, img_w, img_h, class_id)
    if not yolo_lines:
        # No valid annotations — remove copied image and skip
        os.remove(img_dst)
        return False

    with open(lbl_dst, "w") as f:
        f.write("\n".join(yolo_lines) + "\n")

    return True


def verify_parity(split):
    """Check that every image has a matching label and vice-versa."""
    img_dir = os.path.join(DATASET_DIR, "images", split)
    lbl_dir = os.path.join(DATASET_DIR, "labels", split)

    img_stems = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")}
    lbl_stems = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir) if f.endswith(".txt")}

    missing_labels = img_stems - lbl_stems
    missing_images = lbl_stems - img_stems

    if missing_labels:
        print(f"[WARN] {len(missing_labels)} images missing labels in [{split}]")
    if missing_images:
        print(f"[WARN] {len(missing_images)} labels missing images in [{split}]")
    if not missing_labels and not missing_images:
        print(f"[OK] [{split}] parity check passed — {len(img_stems)} image/label pairs.")


def write_data_yaml():
    yaml_path = os.path.join(DATASET_DIR, "..", "data.yaml")
    abs_dataset = os.path.abspath(DATASET_DIR)
    content = f"""# YOLOv11 dataset config
path: {abs_dataset}

train: images/train
val:   images/val

nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        content += f"  {i}: {name}\n"

    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"[OK] data.yaml written to: {yaml_path}")
    return yaml_path



def main():
    random.seed(SEED)
    make_dirs()

    all_train_samples = []
    all_val_samples   = []

    # ── Collect from OID 'train' split ──────────────────────────────
    oid_train = os.path.join(OID_ROOT, "train")
    train_raw = collect_samples(oid_train)

    # ── Collect from OID 'validation' split (optional) ──────────────
    oid_val = os.path.join(OID_ROOT, "validation")
    val_raw = collect_samples(oid_val)

    if val_raw:
        # Use OID's own split
        all_train_samples = train_raw
        all_val_samples   = val_raw
    else:
        # Manual 80/20 split from train only
        print("[INFO] No OID validation split found — performing 80/20 random split.")
        random.shuffle(train_raw)
        cut = int(len(train_raw) * (1 - VAL_RATIO))
        all_train_samples = train_raw[:cut]
        all_val_samples   = train_raw[cut:]

    print(f"\n[INFO] Train samples: {len(all_train_samples)}")
    print(f"[INFO] Val   samples: {len(all_val_samples)}")

    # ── Write train ──────────────────────────────────────────────────
    print("\n[...] Writing TRAIN split ...")
    ok = sum(write_sample(i, l, c, "train") for i, l, c in all_train_samples)
    print(f"[OK]  {ok} / {len(all_train_samples)} train samples written.")

    # ── Write val ────────────────────────────────────────────────────
    print("\n[...] Writing VAL split ...")
    ok = sum(write_sample(i, l, c, "val") for i, l, c in all_val_samples)
    print(f"[OK]  {ok} / {len(all_val_samples)} val samples written.")

    # ── Parity check ─────────────────────────────────────────────────
    print("\n[...] Running parity checks ...")
    verify_parity("train")
    verify_parity("val")

    # ── Write data.yaml ───────────────────────────────────────────────
    yaml_path = write_data_yaml()

    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Dataset ready at : {os.path.abspath(DATASET_DIR)}/")
    print(f"Config yaml      : {yaml_path}")
    print(f"\nNext step: run  python 3_train.py")


if __name__ == "__main__":
    main()