import argparse
import os
import sys
import time

import cv2

DEFAULT_WEIGHTS  = r"C:\Users\lenovo\Desktop\OD\runs\detect\runs\detect\bottle_tincan_dice_ball\weights\best.pt"
FALLBACK_WEIGHTS = "yolo11n.pt"          # Uses pretrained COCO weights as fallback
CONF_THRESHOLD   = 0.40                  # Minimum confidence to display a detection
IOU_THRESHOLD    = 0.45                  # NMS IoU threshold
IMGSZ            = 640

# Class colours: Bottle=blue, Tin can=green, Dice=red, Ball=orange
CLASS_COLORS = {
    0: (230,  80,  80),   # Bottle   — blue  (BGR)
    1: ( 80, 200,  80),   # Tin can  — green
    2: ( 80,  80, 230),   # Dice     — red
    3: ( 50, 165, 255),   # Ball     — orange
}
CLASS_NAMES = ["Bottle", "Tin can", "Dice", "Ball"]


def load_model(weights_path):
    from ultralytics import YOLO
    if not os.path.isfile(weights_path):
        print(f"[WARN] Weights not found at: {weights_path}")
        print(f"[INFO] Falling back to pretrained model: {FALLBACK_WEIGHTS}")
        print("       (This will detect COCO classes, not your custom classes.)")
        print("       Run step 3 to train your own model first.")
        weights_path = FALLBACK_WEIGHTS
    else:
        print(f"[OK] Loaded model: {weights_path}")
    return YOLO(weights_path)


def draw_detections(frame, results, conf_thresh):
    """Draw bounding boxes and labels onto the frame."""
    det_count = 0
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
            text  = f"{label}  {conf:.0%}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

            # Label text
            cv2.putText(
                frame, text,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA
            )
            det_count += 1

    return frame, det_count


def draw_hud(frame, fps, det_count):
    """Draw FPS and detection count overlay."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (220, 60), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}",          (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 1)
    cv2.putText(frame, f"Detections: {det_count}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)
    return frame


def run_webcam(model, conf, save):
    """Live webcam detection loop."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam (index 0).")
        print("Try a different camera index or check permissions.")
        sys.exit(1)

    print("\n[WEBCAM] Starting live detection ...")
    print("  Press  Q  to quit")
    print("  Press  S  to save current frame")

    writer = None
    if save:
        os.makedirs("output", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = "output/webcam_output.mp4"
        writer = cv2.VideoWriter(out_path, fourcc, fps_out, (w, h))
        print(f"  Saving video to: {out_path}")

    t_prev = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        results  = model(frame, conf=conf, iou=IOU_THRESHOLD, imgsz=IMGSZ, verbose=False)
        frame, det_count = draw_detections(frame, results, conf)

        # FPS
        t_now = time.time()
        fps   = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        frame_count += 1

        frame = draw_hud(frame, fps, det_count)

        if writer:
            writer.write(frame)

        cv2.imshow("YOLOv11 — Object Detection  [Q=quit  S=save frame]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            os.makedirs("output", exist_ok=True)
            fname = f"output/frame_{frame_count:05d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[SAVED] {fname}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[DONE] Webcam session ended.")


def run_images(model, source, conf, save):
    """Run detection on a single image or folder of images."""
    import glob as _glob

    if os.path.isdir(source):
        paths = (
            _glob.glob(os.path.join(source, "*.jpg")) +
            _glob.glob(os.path.join(source, "*.jpeg")) +
            _glob.glob(os.path.join(source, "*.png"))
        )
        if not paths:
            print(f"[ERROR] No images found in: {source}")
            sys.exit(1)
        print(f"[INFO] Found {len(paths)} images in: {source}")
    elif os.path.isfile(source):
        paths = [source]
    else:
        print(f"[ERROR] Source not found: {source}")
        sys.exit(1)

    os.makedirs("output", exist_ok=True)

    for img_path in paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[SKIP] Could not read: {img_path}")
            continue

        results = model(frame, conf=conf, iou=IOU_THRESHOLD, imgsz=IMGSZ, verbose=False)
        frame, det_count = draw_detections(frame, results, conf)

        fname = os.path.basename(img_path)
        print(f"  {fname}  →  {det_count} detection(s)")

        if save:
            out_path = os.path.join("output", fname)
            cv2.imwrite(out_path, frame)
        else:
            cv2.imshow(f"Detection — {fname}  [any key=next  Q=quit]", frame)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    if save:
        print(f"[SAVED] Results saved to: output/")
    print("[DONE]")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv11 Object Detector — Bottle, Tin can, Dice, Ball"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Image path, folder path, or leave blank for webcam."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help=f"Path to .pt weights file (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD})"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output images/video instead of displaying live."
    )
    args = parser.parse_args()

    model = load_model(args.weights)

    if args.source is None:
        run_webcam(model, args.conf, args.save)
    else:
        run_images(model, args.source, args.conf, args.save)


if __name__ == "__main__":
    main()
