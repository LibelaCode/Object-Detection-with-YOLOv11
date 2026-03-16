import subprocess
import sys
import os

OIDV4_PATH = "./OIDv4_ToolKit" 
DATASET_OUTPUT = "./OID_raw"           
LIMIT = 500                           
CLASSES = ["Bottle", "Tin_can", "Dice", "Ball"]
SPLITS  = ["train", "validation"]   

def run_download(cls_list, split, limit, dataset_dir):
    """Call OIDv4 downloader for a given split."""
    classes_arg = " ".join(cls_list)
    cmd = (
        f"python {OIDV4_PATH}/main.py downloader "
        f"--classes {classes_arg} "
        f"--type_csv {split} "
        f"--limit {limit} "
        f"--Dataset {dataset_dir} "
        f"--multiclasses 0 "   
        f"-y"                  
    )
    print(f"\n{'='*60}")
    print(f"Downloading [{split}] split for: {cls_list}")
    print(f"Command: {cmd}")
    print("="*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARNING] Download returned non-zero exit code for {split}.")


def verify_oidv4_exists():
    main_py = os.path.join(OIDV4_PATH, "main.py")
    if not os.path.isfile(main_py):
        print(f"[ERROR] OIDv4_ToolKit not found at: {OIDV4_PATH}")
        print("Please clone it first:")
        print("  git clone https://github.com/EscVM/OIDv4_ToolKit.git")
        print("  pip3 install -r OIDv4_ToolKit/requirements.txt")
        sys.exit(1)
    print(f"[OK] OIDv4_ToolKit found at: {OIDV4_PATH}")


def print_summary():
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE — Expected folder structure:")
    print("="*60)
    print(f"{DATASET_OUTPUT}/")
    print("└── OID/")
    print("    └── Dataset/")
    for split in SPLITS:
        print(f"        └── {split}/")
        for cls in CLASSES:
            cls_display = cls.replace("_", " ").title()
            print(f"            └── {cls_display}/")
            print(f"                ├── <image>.jpg  ...")
            print(f"                └── Labels/")
            print(f"                    └── <image>.txt  ...")
    print("\nNext step: run  python 2_convert_labels.py")


def main():
    verify_oidv4_exists()
    os.makedirs(DATASET_OUTPUT, exist_ok=True)

    for split in SPLITS:
        run_download(CLASSES, split, LIMIT, DATASET_OUTPUT)

    print_summary()


if __name__ == "__main__":
    main()
