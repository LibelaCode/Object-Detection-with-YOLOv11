import os
import sys

DATA_YAML   = "./data.yaml"          
MODEL       = "yolo11n.pt"            
EPOCHS      = 150                     
IMGSZ       = 640                     
BATCH       = -1                     
PATIENCE    = 30                     
WORKERS     = 2                       
PROJECT     = "runs/detect"           # Output folder
RUN_NAME    = "bottle_tincan_dice_ball"
DEVICE      = None                    # None = auto (GPU if available, else CPU)
                                      # "cpu" to force CPU, "0" for first GPU


def check_requirements():
    try:
        from ultralytics import YOLO
        print("[OK] ultralytics is installed.")
    except ImportError:
        print("[ERROR] ultralytics not found.")
        print("Install it with:  pip install ultralytics")
        sys.exit(1)

    if not os.path.isfile(DATA_YAML):
        print(f"[ERROR] data.yaml not found at: {DATA_YAML}")
        print("Run step 2 first:  python 2_convert_labels.py")
        sys.exit(1)
    print(f"[OK] data.yaml found: {DATA_YAML}")


def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {gpu}  |  VRAM: {vram:.1f} GB")
            return "0"
        else:
            print("[CPU] No CUDA GPU detected — training on CPU (slow).")
            return "cpu"
    except ImportError:
        print("[CPU] PyTorch not found — defaulting to CPU.")
        return "cpu"

def train():
    from ultralytics import YOLO

    device = DEVICE if DEVICE is not None else detect_device()

    # Adjust batch size automatically for CPU
    batch = BATCH
    if device == "cpu" and BATCH > 8:
        batch = 8
        print(f"[INFO] CPU detected — reducing batch to {batch}.")

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Model     : {MODEL}")
    print(f"  Data      : {DATA_YAML}")
    print(f"  Epochs    : {EPOCHS}  (patience={PATIENCE})")
    print(f"  Img size  : {IMGSZ}")
    print(f"  Batch     : {batch}")
    print(f"  Device    : {device}")
    print(f"  Output    : {PROJECT}/{RUN_NAME}/")
    print("="*60 + "\n")

    model = YOLO(MODEL)  # Downloads pretrained weights automatically

    results = model.train(
        data      = DATA_YAML,
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = batch,
        patience  = PATIENCE,
        workers   = WORKERS,
        device    = device,
        project   = PROJECT,
        name      = RUN_NAME,
        exist_ok  = True,
        verbose   = True,
        amp       = False,
    )

    best_weights = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best weights  : {best_weights}")
    print(f"Results saved : {PROJECT}/{RUN_NAME}/")
    print("\nKey metrics to check:")
    print("  • box_loss  → should decrease steadily")
    print("  • cls_loss  → should drop quickly")
    print("  • mAP50     → should increase over time (target > 0.7)")
    print(f"\nNext step: run  python 4_detect.py")
    return best_weights

def validate(weights_path):
    from ultralytics import YOLO
    print(f"\n[...] Running validation with: {weights_path}")
    model = YOLO(weights_path)
    metrics = model.val(data=DATA_YAML, imgsz=IMGSZ)
    print(f"[OK] mAP50      : {metrics.box.map50:.4f}")
    print(f"[OK] mAP50-95   : {metrics.box.map:.4f}")


def main():
    check_requirements()
    best = train()

    # Uncomment to auto-validate after training:
    # if os.path.isfile(best):
    #     validate(best)


if __name__ == "__main__":
    main()
