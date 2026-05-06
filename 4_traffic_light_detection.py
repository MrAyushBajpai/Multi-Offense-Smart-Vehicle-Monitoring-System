"""
YOLO Traffic Signal Detection for Video Files
Uses Ultralytics YOLO (supports YOLOv8 through YOLO11/v26 via ultralytics package)

Detects and classifies traffic signals:
- Red Light
- Yellow Light
- Green Light

Also detects the traffic light object itself (COCO class 9).

Reads video clips from /Input folder and saves annotated output videos.
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
DEFAULT_VIDEO = "Input/traffic_light.mp4"

# YOLO model
MODEL_WEIGHTS = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD        = 0.45
DEVICE               = ""

SHOW_CONF            = True
LINE_WIDTH           = 2

# COCO class 9 = traffic light
TRAFFIC_LIGHT_CLASS = 9

# HSV colour ranges for red / yellow / green detection
# Red wraps around 0°, so we need two ranges
COLOR_RANGES = {
    "Red": [
        (np.array([0,   120,  70]),  np.array([10,  255, 255])),
        (np.array([170, 120,  70]),  np.array([180, 255, 255])),
    ],
    "Yellow": [
        (np.array([18, 100, 100]),  np.array([35, 255, 255])),
    ],
    "Green": [
        (np.array([36, 80, 80]),   np.array([89, 255, 255])),
    ],
}

# Overlay colours (BGR) per signal state
SIGNAL_COLORS = {
    "Red":     (0,   0,   255),
    "Yellow":  (0,   215, 255),
    "Green":   (0,   200,  0),
    "Unknown": (180, 180, 180),
}


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".mp4", ".avi", ".mov",
    ".mkv", ".wmv", ".flv", ".webm",
}


def get_video_files(folder: str) -> list[Path]:
    """Return all supported video files found in folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return [
        p for p in sorted(folder_path.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_model(weights: str) -> YOLO:
    """Load YOLO model with fallback support."""
    try:
        model = YOLO(weights)
        print(f"[INFO] Loaded model: {weights}")
        return model
    except Exception as e:
        fallback = "yolov8n.pt"
        print(f"[WARN] Could not load '{weights}': {e}")
        print(f"[INFO] Falling back to '{fallback}'")
        return YOLO(fallback)


def classify_signal_color(roi: np.ndarray) -> str:
    """
    Determine traffic-light state (Red / Yellow / Green / Unknown)
    from the cropped bounding-box region using HSV colour analysis.
    """
    if roi is None or roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    pixel_counts = {}

    for color_name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        pixel_counts[color_name] = int(np.sum(mask > 0))

    best_color = max(pixel_counts, key=pixel_counts.get)

    # Require at least 1 % of the ROI to be the dominant colour
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0 or pixel_counts[best_color] < total_pixels * 0.01:
        return "Unknown"

    return best_color


def draw_signal_counts(frame, counts: dict) -> None:
    """Draw per-frame signal state counts in the top-left corner."""
    y = 30
    for label, count in counts.items():
        color = SIGNAL_COLORS.get(label, (255, 255, 255))
        text  = f"{label}: {count}"
        cv2.putText(
            frame, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
        y += 30


def process_video(model: YOLO, video_path: Path, output_folder: str) -> None:
    """Run traffic-signal detection on a video and save annotated output."""

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
        f"{video_path.stem}_signals{video_path.suffix}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx  = 0
    total_dets = 0

    # Running totals across the whole video
    global_counts = {"Red": 0, "Yellow": 0, "Green": 0, "Unknown": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        result    = results[0]
        annotated = frame.copy()

        frame_counts = {"Red": 0, "Yellow": 0, "Green": 0, "Unknown": 0}

        for box in result.boxes:
            cls_id = int(box.cls[0])

            if cls_id != TRAFFIC_LIGHT_CLASS:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clamp coordinates to frame boundaries
            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(width,  x2); y2c = min(height, y2)

            roi         = frame[y1c:y2c, x1c:x2c]
            signal_state = classify_signal_color(roi)

            frame_counts[signal_state]  += 1
            global_counts[signal_state] += 1
            total_dets += 1

            box_color = SIGNAL_COLORS[signal_state]

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, LINE_WIDTH)

            # Label
            label = f"Signal:{signal_state}"
            if SHOW_CONF:
                label += f" {confidence:.2f}"

            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
            )

            cv2.rectangle(
                annotated,
                (x1, y1 - 30),
                (x1 + text_w + 10, y1),
                box_color,
                -1,
            )

            cv2.putText(
                annotated, label, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
            )

        # ── Overlay info ───────────────────────────────────────────
        cv2.putText(
            annotated,
            f"Frame {frame_idx+1}/{total}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        draw_signal_counts(annotated, frame_counts)

        writer.write(annotated)

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(
                f"  → Frame {frame_idx}/{total} processed "
                f"(signal detections so far: {total_dets})"
            )

    cap.release()
    writer.release()

    print(
        f"[INFO] Done. "
        f"{frame_idx} frames processed, "
        f"{total_dets} signal detections."
    )
    print(f"[INFO] Output saved to: {out_path}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n  Traffic Signal Summary (whole video):")
    for state, count in global_counts.items():
        if count > 0:
            print(f"    {state:<10} {count:>6} detections")
    if total_dets == 0:
        print("    No traffic signals detected.")


def print_detection_summary(model: YOLO, video_path: Path) -> None:
    """Print total traffic-signal colour counts for a video."""

    print(f"\n[INFO] Generating signal summary for: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    counts = {"Red": 0, "Yellow": 0, "Green": 0, "Unknown": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id != TRAFFIC_LIGHT_CLASS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            state = classify_signal_color(roi)
            counts[state] += 1

    cap.release()

    print("\n  Traffic Signal Detection Summary:")
    found = False
    for state, count in counts.items():
        if count > 0:
            found = True
            print(f"    {state:<10} {count:>6}")
    if not found:
        print("    No traffic signals detected.")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
def main():

    global CONFIDENCE_THRESHOLD, MODEL_WEIGHTS

    parser = argparse.ArgumentParser(
        description="YOLO Traffic Signal Detection on video files."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_VIDEO,
        help=f"Path to video OR folder of videos. Default: '{DEFAULT_VIDEO}'",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FOLDER,
        help=f"Output folder (default: {OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--model",
        default=MODEL_WEIGHTS,
        help=f"YOLO weights file (default: {MODEL_WEIGHTS})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print traffic signal detection summary after processing.",
    )

    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf
    MODEL_WEIGHTS        = args.model

    model = load_model(args.model)

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
        process_video(model, video, args.output)
        if args.summary:
            print_detection_summary(model, video)

    print(f"\n[INFO] All done. Annotated videos are in: {args.output}")


if __name__ == "__main__":
    main()