"""
YOLO Vehicle Classification for Video Files
Uses Ultralytics YOLO (supports YOLOv8 through YOLO11/v26 via ultralytics package)

Detects and classifies:
- Car
- Motorcycle
- Bus
- Truck

Reads video clips from /Input folder and saves annotated output videos.
"""

import cv2
import os
import argparse
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
INPUT_FOLDER = "Input"
OUTPUT_FOLDER = "Output"
DEFAULT_VIDEO = "Input/input.mp4"

# YOLO model
MODEL_WEIGHTS = "yolo26m.pt"

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD        = 0.45
DEVICE               = ""

SHOW_LABELS          = True
SHOW_CONF            = True
LINE_WIDTH           = 2

# Vehicle classes from COCO dataset
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".mp4", ".avi", ".mov",
    ".mkv", ".wmv", ".flv", ".webm"
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


def draw_vehicle_counts(frame, counts: dict) -> None:
    """Draw vehicle classification counts on frame."""

    y = 30

    for label, count in counts.items():
        text = f"{label}: {count}"

        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        y += 30


def process_video(model: YOLO, video_path: Path, output_folder: str) -> None:
    """Run vehicle classification on video and save annotated output."""

    print(f"\n{'='*60}")
    print(f"[INFO] Processing: {video_path.name}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    # Video properties
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"[INFO] Resolution: {width}x{height}  |  "
        f"FPS: {fps:.1f}  |  Frames: {total}"
    )

    # Output path
    os.makedirs(output_folder, exist_ok=True)

    out_path = (
        Path(output_folder) /
        f"{video_path.stem}_classified{video_path.suffix}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        fps,
        (width, height)
    )

    frame_idx  = 0
    total_dets = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # ── Run inference ──────────────────────────────────────────
        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]

        annotated = frame.copy()

        # Per-frame counts
        vehicle_counts = {
            "Car": 0,
            "Motorcycle": 0,
            "Bus": 0,
            "Truck": 0,
        }

        # ── Process detections ─────────────────────────────────────
        for box in result.boxes:

            cls_id = int(box.cls[0])

            # Ignore non-vehicle classes
            if cls_id not in VEHICLE_CLASSES:
                continue

            confidence = float(box.conf[0])

            vehicle_name = VEHICLE_CLASSES[cls_id]

            vehicle_counts[vehicle_name] += 1
            total_dets += 1

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                LINE_WIDTH,
            )

            # Label text
            if SHOW_CONF:
                label = f"{vehicle_name} {confidence:.2f}"
            else:
                label = vehicle_name

            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )

            cv2.rectangle(
                annotated,
                (x1, y1 - 30),
                (x1 + text_w + 10, y1),
                (0, 255, 0),
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # ── Draw overlay info ──────────────────────────────────────
        cv2.putText(
            annotated,
            f"Frame {frame_idx+1}/{total}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        draw_vehicle_counts(annotated, vehicle_counts)

        # ── Save frame ─────────────────────────────────────────────
        writer.write(annotated)

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(
                f"  → Frame {frame_idx}/{total} processed "
                f"(vehicle detections so far: {total_dets})"
            )

    cap.release()
    writer.release()

    print(
        f"[INFO] Done. "
        f"{frame_idx} frames processed, "
        f"{total_dets} vehicle detections."
    )

    print(f"[INFO] Output saved to: {out_path}")


def print_detection_summary(model: YOLO, video_path: Path) -> None:
    """Print total vehicle classification counts."""

    print(f"\n[INFO] Generating vehicle summary for: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    class_counts = {
        "Car": 0,
        "Motorcycle": 0,
        "Bus": 0,
        "Truck": 0,
    }

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        for box in results[0].boxes:

            cls_id = int(box.cls[0])

            if cls_id not in VEHICLE_CLASSES:
                continue

            cls_name = VEHICLE_CLASSES[cls_id]

            class_counts[cls_name] += 1

    cap.release()

    print("\n  Vehicle Detection Summary:")

    found = False

    for cls, count in class_counts.items():

        if count > 0:
            found = True
            print(f"    {cls:<15} {count:>6}")

    if not found:
        print("    No vehicles detected.")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
def main():

    global CONFIDENCE_THRESHOLD, MODEL_WEIGHTS

    parser = argparse.ArgumentParser(
        description="YOLO Vehicle Classification on video files."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_VIDEO,
        help=(
            f"Path to video OR folder of videos. "
            f"Default: '{DEFAULT_VIDEO}'"
        ),
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
        help="Print vehicle classification summary.",
    )

    args = parser.parse_args()

    # Override globals
    CONFIDENCE_THRESHOLD = args.conf
    MODEL_WEIGHTS        = args.model

    # Load model
    model = load_model(args.model)

    # Single file or folder
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

    # Process videos
    for video in videos:

        process_video(model, video, args.output)

        if args.summary:
            print_detection_summary(model, video)

    print(f"\n[INFO] All done. Classified videos are in: {args.output}")


if __name__ == "__main__":
    main()