"""
YOLO Seatbelt Violation Detection for Video Files
Uses Ultralytics YOLO (supports YOLOv8 through YOLO11 via ultralytics package)

Detection logic:
  1. Detect all vehicles (car, bus, truck) in the frame.
  2. For each vehicle, detect persons inside (overlapping bounding box).
  3. For each occupant, check whether a seatbelt is present.
  4. Flag as VIOLATION if an occupant's torso region shows no seatbelt.

Seatbelt detection strategy (two-tier):
  - Primary  : dedicated seatbelt classifier model (e.g. seatbelt_best.pt) if supplied
  - Fallback : torso-region diagonal-line heuristic using Canny + Hough lines

Reads video clips from /Input folder and saves annotated output videos.

Usage examples
--------------
# Default  (Input/input.mp4)
python seatbelt_detection.py

# Custom video
python seatbelt_detection.py --input Input/dashcam.mp4

# Whole folder
python seatbelt_detection.py --input Input/

# Dedicated seatbelt model + summary
python seatbelt_detection.py --seatbelt-model seatbelt_best.pt --summary

# Lower confidence
python seatbelt_detection.py --conf 0.35
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
DEFAULT_VIDEO = "Input/seatbelt_1.mp4"

BASE_MODEL_WEIGHTS     = "yolo11n.pt"   # general COCO detector
SEATBELT_MODEL_WEIGHTS = ""             # dedicated seatbelt model; "" = use heuristic

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD        = 0.45
DEVICE               = ""

SHOW_CONF  = True
LINE_WIDTH = 2

# ── COCO class IDs ──────────────────────────────────────────────
PERSON_CLASS = 0
VEHICLE_CLASSES = {2, 5, 7}   # car=2, bus=5, truck=7

# Dedicated model class IDs (adjust to match your model)
SEATBELT_CLASS_ID    = 0   # "seatbelt"    / "with_seatbelt"
NO_SEATBELT_CLASS_ID = 1   # "no_seatbelt" / "without_seatbelt"

# Minimum overlap to associate a person with a vehicle
OCCUPANT_IOU_THRESHOLD = 0.10

# Torso region: vertical slice of the person box used for seatbelt check
# (skip top 20 % head, use next 55 % = chest/shoulder area)
TORSO_TOP_FRAC    = 0.20
TORSO_BOTTOM_FRAC = 0.75

# ── Visual colours (BGR) ────────────────────────────────────────
COLOR_SAFE      = (0,   200,   0)    # green  – seatbelt on
COLOR_VIOLATION = (0,     0, 255)    # red    – no seatbelt
COLOR_VEHICLE   = (255, 165,   0)    # orange – vehicle box
COLOR_UNKNOWN   = (180, 180, 180)    # grey


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
        print(f"[INFO] Seatbelt model unavailable – using heuristic fallback.")
        return None


def box_iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)


def get_torso_roi(frame, person_box):
    """
    Extract the chest/shoulder region from a person bounding box.
    Skips the head (top 20%) and crops to the upper torso (next 55%).
    Returns (roi, (rx1, ry1, rx2, ry2)) or None.
    """
    x1, y1, x2, y2 = person_box
    h_box = y2 - y1
    ry1 = y1 + int(h_box * TORSO_TOP_FRAC)
    ry2 = y1 + int(h_box * TORSO_BOTTOM_FRAC)
    fh, fw = frame.shape[:2]
    rx1 = max(0, x1); ry1 = max(0, ry1)
    rx2 = min(fw, x2); ry2 = min(fh, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return frame[ry1:ry2, rx1:rx2], (rx1, ry1, rx2, ry2)


# ── Heuristic seatbelt check ────────────────────────────────────
def heuristic_seatbelt_check(torso_roi: np.ndarray) -> bool:
    """
    Detects a seatbelt by looking for a prominent diagonal line
    (the shoulder-to-lap strap) in the torso region.

    A seatbelt strap:
      - Is a near-vertical or diagonal line (30°–80° from horizontal)
      - Is relatively narrow but high-contrast against clothing
      - Appears in the upper torso area

    Returns True if a seatbelt-like strap is likely present.
    """
    if torso_roi is None or torso_roi.size == 0:
        return False

    h, w = torso_roi.shape[:2]

    # ── Step 1: edge detection ──────────────────────────────────
    gray    = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)

    # ── Step 2: Hough line detection ────────────────────────────
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(20, h // 4),   # scale threshold with ROI height
        minLineLength=max(15, h // 3),
        maxLineGap=10,
    )

    if lines is None:
        return False

    diagonal_count = 0

    for line in lines:
        x1l, y1l, x2l, y2l = line[0]

        dx = x2l - x1l
        dy = y2l - y1l

        if dx == 0:
            angle_deg = 90.0
        else:
            angle_deg = abs(np.degrees(np.arctan2(dy, dx)))

        # Seatbelt strap runs roughly 30°–80° from horizontal
        # (diagonal across the chest, not purely horizontal/vertical)
        if 25 <= angle_deg <= 80:
            line_len = np.hypot(dx, dy)
            # Must span at least 1/3 of the ROI height to be significant
            if line_len >= h * 0.33:
                diagonal_count += 1

    # ── Step 3: colour contrast check ───────────────────────────
    # Seatbelt straps often appear as a narrow band with distinct
    # brightness contrast against the clothing underneath.
    # Check for a significant vertical brightness gradient.
    gray_norm = gray.astype(np.float32) / 255.0
    col_std   = np.std(gray_norm, axis=0)   # std per column
    high_contrast_cols = int(np.sum(col_std > 0.12))
    contrast_ratio = high_contrast_cols / max(w, 1)

    # Verdict: diagonal line present OR high column-wise contrast
    return diagonal_count >= 1 or contrast_ratio > 0.35


# ── Model-based seatbelt check ───────────────────────────────────
def model_seatbelt_check(
    seatbelt_model: YOLO,
    torso_roi: np.ndarray,
    conf: float,
) -> bool | None:
    """
    Returns True  if seatbelt detected,
            False if no-seatbelt detected,
            None  if no relevant detection (fall through to heuristic).
    """
    if torso_roi is None or torso_roi.size == 0:
        return None

    results = seatbelt_model(torso_roi, conf=conf, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == SEATBELT_CLASS_ID:
            return True
        if cls_id == NO_SEATBELT_CLASS_ID:
            return False
    return None


def has_seatbelt(
    frame: np.ndarray,
    person_box: tuple,
    seatbelt_model: YOLO | None,
    conf: float,
) -> tuple[bool, tuple]:
    """
    Determine seatbelt presence for an occupant.
    Returns (seatbelt_present, torso_box_coords).
    """
    result = get_torso_roi(frame, person_box)
    if result is None:
        return False, (0, 0, 0, 0)

    torso_roi, torso_coords = result

    if seatbelt_model is not None:
        decision = model_seatbelt_check(seatbelt_model, torso_roi, conf)
        if decision is not None:
            return decision, torso_coords

    return heuristic_seatbelt_check(torso_roi), torso_coords


# ─────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────
def process_video(
    base_model: YOLO,
    seatbelt_model: YOLO | None,
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

    out_path = (
        Path(output_folder) /
        f"{video_path.stem}_seatbelt_check{video_path.suffix}"
    )
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx        = 0
    total_occupants  = 0
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

        boxes     = results[0].boxes
        annotated = frame.copy()

        persons  = []    # (box_xyxy, confidence)
        vehicles = []    # (box_xyxy, confidence, class_id)

        for box in boxes:
            cls_id = int(box.cls[0])
            coords = tuple(map(int, box.xyxy[0]))
            conf   = float(box.conf[0])

            if cls_id == PERSON_CLASS:
                persons.append((coords, conf))
            elif cls_id in VEHICLE_CLASSES:
                vehicles.append((coords, conf, cls_id))

        # ── Draw vehicle boxes ──────────────────────────────────
        vehicle_labels = {2: "Car", 5: "Bus", 7: "Truck"}
        for (v_box, v_conf, v_cls) in vehicles:
            x1, y1, x2, y2 = v_box
            label = vehicle_labels.get(v_cls, "Vehicle")
            if SHOW_CONF:
                label += f" {v_conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_VEHICLE, LINE_WIDTH)
            _draw_label(annotated, label, x1, y1, COLOR_VEHICLE)

        # ── Associate occupants with vehicles ───────────────────
        frame_violations = 0
        frame_compliant  = 0
        occupant_set     = set()

        for (v_box, _, _) in vehicles:
            for p_idx, (p_box, p_conf) in enumerate(persons):
                if box_iou(v_box, p_box) < OCCUPANT_IOU_THRESHOLD:
                    continue
                if p_idx in occupant_set:
                    continue
                occupant_set.add(p_idx)

                belt_on, torso_coords = has_seatbelt(
                    frame, p_box, seatbelt_model, CONFIDENCE_THRESHOLD
                )

                total_occupants += 1
                px1, py1, px2, py2 = p_box
                tx1, ty1, tx2, ty2 = torso_coords

                if belt_on:
                    total_compliant  += 1
                    frame_compliant  += 1
                    p_color = COLOR_SAFE
                    status  = "Seatbelt ON"
                else:
                    total_violations += 1
                    frame_violations += 1
                    p_color = COLOR_VIOLATION
                    status  = "NO SEATBELT"

                # Person bounding box
                cv2.rectangle(annotated, (px1, py1), (px2, py2), p_color, LINE_WIDTH)
                p_label = status + (f" {p_conf:.2f}" if SHOW_CONF else "")
                _draw_label(annotated, p_label, px1, py1, p_color)

                # Torso ROI box (thin border)
                if tx2 > tx1 and ty2 > ty1:
                    cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), p_color, 1)

        # ── HUD overlay ─────────────────────────────────────────
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
    _print_summary(total_occupants, total_violations, total_compliant)


# ─────────────────────────────────────────
# Drawing utilities
# ─────────────────────────────────────────
def _draw_label(frame, text: str, x1: int, y1: int, color: tuple) -> None:
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
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
    cv2.putText(
        frame, f"Frame {frame_idx+1}/{total}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
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


def _print_summary(occupants: int, violations: int, compliant: int) -> None:
    print("\n  ── Seatbelt Violation Summary ────────────────")
    print(f"    Total occupants detected : {occupants}")
    print(f"    Compliant (belt on)      : {compliant}")
    print(f"    VIOLATIONS (no belt)     : {violations}")
    if occupants > 0:
        pct = violations / occupants * 100
        print(f"    Violation rate           : {pct:.1f}%")
    print("  ──────────────────────────────────────────────")


def print_detection_summary(
    base_model: YOLO,
    seatbelt_model: YOLO | None,
    video_path: Path,
) -> None:
    print(f"\n[INFO] Generating seatbelt summary for: {video_path.name}")

    cap    = cv2.VideoCapture(str(video_path))
    fw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    occupants = violations = compliant = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results  = base_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes    = results[0].boxes

        persons  = [(tuple(map(int, b.xyxy[0])), float(b.conf[0]))
                    for b in boxes if int(b.cls[0]) == PERSON_CLASS]
        vehicles = [(tuple(map(int, b.xyxy[0])),)
                    for b in boxes if int(b.cls[0]) in VEHICLE_CLASSES]

        seen = set()
        for (v_box,) in vehicles:
            for i, (p_box, _) in enumerate(persons):
                if box_iou(v_box, p_box) < OCCUPANT_IOU_THRESHOLD or i in seen:
                    continue
                seen.add(i)
                occupants += 1
                belt_on, _ = has_seatbelt(
                    frame, p_box, seatbelt_model, CONFIDENCE_THRESHOLD
                )
                if belt_on:
                    compliant += 1
                else:
                    violations += 1

    cap.release()
    _print_summary(occupants, violations, compliant)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
def main():

    global CONFIDENCE_THRESHOLD, BASE_MODEL_WEIGHTS, SEATBELT_MODEL_WEIGHTS

    parser = argparse.ArgumentParser(
        description="YOLO Seatbelt Violation Detection on video files."
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
        "--seatbelt-model", default=SEATBELT_MODEL_WEIGHTS,
        dest="seatbelt_model",
        help="Optional dedicated seatbelt classifier weights (e.g. seatbelt_best.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print seatbelt violation summary after processing.",
    )

    args = parser.parse_args()

    CONFIDENCE_THRESHOLD   = args.conf
    BASE_MODEL_WEIGHTS     = args.model
    SEATBELT_MODEL_WEIGHTS = args.seatbelt_model

    base_model     = load_model(args.model,          label="base model")
    seatbelt_model = load_model(args.seatbelt_model, label="seatbelt model")

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
        process_video(base_model, seatbelt_model, video, args.output)
        if args.summary:
            print_detection_summary(base_model, seatbelt_model, video)

    print(f"\n[INFO] All done. Annotated videos are in: {args.output}")


if __name__ == "__main__":
    main()