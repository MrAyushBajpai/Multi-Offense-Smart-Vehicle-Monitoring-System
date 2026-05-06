"""
YOLO Bike Helmet Violation Detection for Video Files
Uses Ultralytics YOLO (supports YOLOv8 through YOLO11 via ultralytics package)

Detection logic:
  1. Detect all motorcycles / bicycles in the frame.
  2. For each two-wheeler, look for a person riding it (overlapping bounding box).
  3. For each rider, check whether a helmet is present on their head region.
  4. Flag as VIOLATION if a rider's head region contains no helmet.

Helmet detection strategy (two-tier):
  - Primary  : dedicated helmet classifier model (e.g. helmet_yolov8.pt) if available
  - Fallback : head-region HSV + shape heuristic (round, high-contrast object)

Reads video clips from /Input folder and saves annotated output videos.

Usage examples
--------------
# Default (Input/input.mp4)
python helmet_violation_detection.py

# Custom video
python helmet_violation_detection.py --input Input/bike_footage.mp4

# Whole folder
python helmet_violation_detection.py --input Input/

# Use a dedicated helmet model and print summary
python helmet_violation_detection.py --helmet-model helmet_yolov8.pt --summary

# Lower confidence threshold
python helmet_violation_detection.py --conf 0.35
"""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
INPUT_FOLDER  = "Input"
OUTPUT_FOLDER = "Output"
DEFAULT_VIDEO = "Input/bike_1.mp4"

BASE_MODEL_WEIGHTS   = "yolo11n.pt"   # general object detector
HELMET_MODEL_WEIGHTS = "best.pt"             # optional dedicated helmet model; leave "" to use heuristic

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD        = 0.45
DEVICE               = ""

SHOW_CONF  = True
LINE_WIDTH = 2

# ── COCO class IDs ─────────────────────────────────────────────
PERSON_CLASS     = 0
BICYCLE_CLASS    = 1
MOTORCYCLE_CLASS = 3

TWO_WHEELER_CLASSES = {BICYCLE_CLASS, MOTORCYCLE_CLASS}

# If a dedicated helmet model is used, set the class IDs it uses:
HELMET_CLASS_ID    = 0   # "helmet"     class in custom model
NO_HELMET_CLASS_ID = 1   # "no_helmet"  class in custom model (optional)

# Overlap IoU needed to associate a person with a two-wheeler
RIDER_IOU_THRESHOLD = 0.10

# Head region: top fraction of the person bounding box used for helmet check
HEAD_FRACTION = 0.30

# ── Visual colours (BGR) ────────────────────────────────────────
COLOR_SAFE      = (0,   200,   0)    # green  – helmet on
COLOR_VIOLATION = (0,   0,   255)    # red    – no helmet
COLOR_TWOWHEELER = (255, 165,   0)   # orange – bike/motorcycle box
COLOR_UNKNOWN   = (180, 180, 180)    # grey   – cannot determine


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".mp4", ".avi", ".mov",
    ".mkv", ".wmv", ".flv", ".webm",
}


def get_video_files(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return [
        p for p in sorted(folder_path.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_model(weights: str, label: str = "model") -> YOLO | None:
    if not weights:
        return None
    try:
        m = YOLO(weights)
        print(f"[INFO] Loaded {label}: {weights}")
        return m
    except Exception as e:
        fallback = "yolov8n.pt"
        print(f"[WARN] Could not load '{weights}': {e}")
        if label == "base model":
            print(f"[INFO] Falling back to '{fallback}'")
            return YOLO(fallback)
        print(f"[INFO] Helmet model unavailable – using heuristic fallback.")
        return None


def box_iou(b1, b2) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


def get_head_roi(frame, person_box, frac: float = HEAD_FRACTION):
    """Return the top-frac of a person bounding box as the head region."""
    x1, y1, x2, y2 = person_box
    head_y2 = y1 + int((y2 - y1) * frac)
    h, w = frame.shape[:2]
    rx1 = max(0, x1); ry1 = max(0, y1)
    rx2 = min(w, x2); ry2 = min(h, head_y2)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return frame[ry1:ry2, rx1:rx2], (rx1, ry1, rx2, ry2)


# ── Heuristic helmet check (no dedicated model) ─────────────────
def heuristic_helmet_check(head_roi: np.ndarray) -> bool:
    """
    Simple heuristic: looks for a rounded, relatively dark or distinctly
    coloured blob in the head ROI.

    Returns True  if a helmet-like object is likely present.
    Returns False if the head region looks bare (skin-dominant).
    """
    if head_roi is None or head_roi.size == 0:
        return False

    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

    # Skin colour mask (bare head)
    skin_lo = np.array([0,  20, 70],  dtype=np.uint8)
    skin_hi = np.array([25, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, skin_lo, skin_hi)

    total   = head_roi.shape[0] * head_roi.shape[1]
    skin_px = int(np.sum(skin_mask > 0))

    # If >40 % of head ROI is skin-coloured → likely no helmet
    skin_ratio = skin_px / max(total, 1)
    if skin_ratio > 0.40:
        return False

    # Check for a helmet-like circular / oval shape via Hough circles
    gray    = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    min_r   = max(5, head_roi.shape[1] // 6)
    max_r   = max(min_r + 1, int(head_roi.shape[1] * 0.8))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=head_roi.shape[0],
        param1=50,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is not None:
        return True

    # Fallback: non-skin, high-saturation pixels (coloured helmet)
    colored_mask = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([180, 255, 255]))
    color_ratio  = int(np.sum(colored_mask > 0)) / max(total, 1)
    return color_ratio > 0.25


# ── Model-based helmet check ─────────────────────────────────────
def model_helmet_check(
    helmet_model: YOLO,
    head_roi: np.ndarray,
    conf: float,
) -> bool | None:
    """
    Returns True  if helmet detected,
            False if no-helmet detected,
            None  if no relevant detection.
    """
    if head_roi is None or head_roi.size == 0:
        return None

    results = helmet_model(head_roi, conf=conf, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == HELMET_CLASS_ID:
            return True
        if cls_id == NO_HELMET_CLASS_ID:
            return False
    return None


def has_helmet(
    frame: np.ndarray,
    person_box: tuple,
    helmet_model: YOLO | None,
    conf: float,
) -> tuple[bool, tuple]:
    """
    Determine helmet presence for a rider.
    Returns (helmet_present, head_box_coords).
    """
    result = get_head_roi(frame, person_box)
    if result is None:
        return False, (0, 0, 0, 0)

    head_roi, head_coords = result

    if helmet_model is not None:
        decision = model_helmet_check(helmet_model, head_roi, conf)
        if decision is not None:
            return decision, head_coords
        # Model gave no verdict – fall through to heuristic
    return heuristic_helmet_check(head_roi), head_coords


# ─────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────
def process_video(
    base_model: YOLO,
    helmet_model: YOLO | None,
    video_path: Path,
    output_folder: str,
) -> None:

    print(f"\n{'='*60}")
    print(f"[INFO] Processing: {video_path.name}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"[INFO] Resolution: {width}x{height}  |  "
        f"FPS: {fps:.1f}  |  Frames: {total}"
    )

    os.makedirs(output_folder, exist_ok=True)

    out_path = Path(output_folder) / f"{video_path.stem}_helmet_check{video_path.suffix}"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0

    # Cumulative counters
    total_riders     = 0
    total_violations = 0
    total_compliant  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = base_model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        boxes  = results[0].boxes
        annotated = frame.copy()

        # Separate detections by type
        persons     = []   # (box_xyxy, confidence)
        two_wheelers = []  # (box_xyxy, confidence, class_id)

        for box in boxes:
            cls_id = int(box.cls[0])
            coords = tuple(map(int, box.xyxy[0]))
            conf   = float(box.conf[0])

            if cls_id == PERSON_CLASS:
                persons.append((coords, conf))
            elif cls_id in TWO_WHEELER_CLASSES:
                two_wheelers.append((coords, conf, cls_id))

        # Draw two-wheeler boxes
        for (tw_box, tw_conf, tw_cls) in two_wheelers:
            x1, y1, x2, y2 = tw_box
            label = ("Motorcycle" if tw_cls == MOTORCYCLE_CLASS else "Bicycle")
            if SHOW_CONF:
                label += f" {tw_conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_TWOWHEELER, LINE_WIDTH)
            _draw_label(annotated, label, x1, y1, COLOR_TWOWHEELER)

        # Associate riders with two-wheelers
        frame_violations = 0
        frame_compliant  = 0
        rider_set        = set()   # person indices already tagged

        for (tw_box, _, _) in two_wheelers:
            for p_idx, (p_box, p_conf) in enumerate(persons):
                if box_iou(tw_box, p_box) < RIDER_IOU_THRESHOLD:
                    continue
                if p_idx in rider_set:
                    continue
                rider_set.add(p_idx)

                helmet_on, head_coords = has_helmet(
                    frame, p_box, helmet_model, CONFIDENCE_THRESHOLD
                )

                total_riders += 1
                px1, py1, px2, py2 = p_box
                hx1, hy1, hx2, hy2 = head_coords

                if helmet_on:
                    total_compliant += 1
                    frame_compliant += 1
                    p_color = COLOR_SAFE
                    status  = "Helmet ON"
                else:
                    total_violations += 1
                    frame_violations += 1
                    p_color = COLOR_VIOLATION
                    status  = "NO HELMET"

                # Person box
                cv2.rectangle(annotated, (px1, py1), (px2, py2), p_color, LINE_WIDTH)
                p_label = status
                if SHOW_CONF:
                    p_label += f" {p_conf:.2f}"
                _draw_label(annotated, p_label, px1, py1, p_color)

                # Head region box
                if hx2 > hx1 and hy2 > hy1:
                    cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), p_color, 1)

        # ── Overlay HUD ────────────────────────────────────────────
        _draw_hud(
            annotated, frame_idx, total,
            frame_violations, frame_compliant, height,
        )

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(
                f"  → Frame {frame_idx}/{total}  |  "
                f"Violations: {total_violations}  Compliant: {total_compliant}"
            )

    cap.release()
    writer.release()

    print(f"\n[INFO] Done. {frame_idx} frames processed.")
    print(f"[INFO] Output saved to: {out_path}")
    _print_summary(total_riders, total_violations, total_compliant)


# ─────────────────────────────────────────
# Drawing utilities
# ─────────────────────────────────────────
def _draw_label(frame, text: str, x1: int, y1: int, color: tuple) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - 28), (x1 + tw + 10, y1), color, -1)
    cv2.putText(
        frame, text, (x1 + 5, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
    )


def _draw_hud(
    frame,
    frame_idx: int,
    total: int,
    violations: int,
    compliant: int,
    height: int,
) -> None:
    # Frame counter
    cv2.putText(
        frame, f"Frame {frame_idx+1}/{total}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    # Counts
    y = 30
    for text, color in [
        (f"Violations : {violations}", COLOR_VIOLATION),
        (f"Compliant  : {compliant}",  COLOR_SAFE),
    ]:
        cv2.putText(
            frame, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
        y += 30


def _print_summary(riders: int, violations: int, compliant: int) -> None:
    print("\n  ── Helmet Violation Summary ──────────────────")
    print(f"    Total riders detected : {riders}")
    print(f"    Compliant (helmet on) : {compliant}")
    print(f"    VIOLATIONS (no helmet): {violations}")
    if riders > 0:
        pct = violations / riders * 100
        print(f"    Violation rate        : {pct:.1f}%")
    print("  ──────────────────────────────────────────────")


def print_detection_summary(
    base_model: YOLO,
    helmet_model: YOLO | None,
    video_path: Path,
) -> None:
    print(f"\n[INFO] Generating helmet summary for: {video_path.name}")

    cap    = cv2.VideoCapture(str(video_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    riders = violations = compliant = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = base_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes   = results[0].boxes

        persons      = [(tuple(map(int, b.xyxy[0])), float(b.conf[0]))
                        for b in boxes if int(b.cls[0]) == PERSON_CLASS]
        two_wheelers = [(tuple(map(int, b.xyxy[0])),)
                        for b in boxes if int(b.cls[0]) in TWO_WHEELER_CLASSES]

        seen = set()
        for (tw_box,) in two_wheelers:
            for i, (p_box, _) in enumerate(persons):
                if box_iou(tw_box, p_box) < RIDER_IOU_THRESHOLD or i in seen:
                    continue
                seen.add(i)
                riders += 1
                helmet_on, _ = has_helmet(frame, p_box, helmet_model, CONFIDENCE_THRESHOLD)
                if helmet_on:
                    compliant += 1
                else:
                    violations += 1

    cap.release()
    _print_summary(riders, violations, compliant)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
def main():

    global CONFIDENCE_THRESHOLD, BASE_MODEL_WEIGHTS, HELMET_MODEL_WEIGHTS

    parser = argparse.ArgumentParser(
        description="YOLO Bike Helmet Violation Detection on video files."
    )
    parser.add_argument(
        "--input", default=DEFAULT_VIDEO,
        help=f"Path to video OR folder of videos. Default: '{DEFAULT_VIDEO}'",
    )
    parser.add_argument(
        "--output", default=OUTPUT_FOLDER,
        help=f"Output folder (default: {OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--model", default=BASE_MODEL_WEIGHTS,
        help=f"Base YOLO weights (default: {BASE_MODEL_WEIGHTS})",
    )
    parser.add_argument(
        "--helmet-model", default=HELMET_MODEL_WEIGHTS,
        dest="helmet_model",
        help="Optional dedicated helmet classifier weights (e.g. helmet_yolov8.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print helmet violation summary after processing.",
    )

    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf
    BASE_MODEL_WEIGHTS   = args.model
    HELMET_MODEL_WEIGHTS = args.helmet_model

    base_model   = load_model(args.model,        label="base model")
    helmet_model = load_model(args.helmet_model, label="helmet model")

    input_path = Path(args.input)

    if input_path.is_dir():
        videos = get_video_files(str(input_path))
        if not videos:
            print(f"[WARN] No supported video files found in: {input_path}")
            return
        print(f"[INFO] Found {len(videos)} video(s) in {input_path}")
    else:
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_path}")
            return
        videos = [input_path]

    for video in videos:
        process_video(base_model, helmet_model, video, args.output)
        if args.summary:
            print_detection_summary(base_model, helmet_model, video)

    print(f"\n[INFO] All done. Annotated videos are in: {args.output}")


if __name__ == "__main__":
    main()