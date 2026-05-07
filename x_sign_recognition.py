"""
YOLO Sign Detection for Video Files
Uses custom trained YOLO11 model: sign.pt

Reads video clips from /Input folder
Saves annotated output videos to /Output folder
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
DEFAULT_VIDEO = "Input/input1.mp4"

# Custom trained model
MODEL_WEIGHTS = "sign.pt"

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD        = 0.45
DEVICE               = ""

SHOW_LABELS          = True
SHOW_CONF            = True
LINE_WIDTH           = 2


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
    """Load YOLO model."""

    try:
        model = YOLO(weights)

        print(f"[INFO] Loaded model: {weights}")

        return model

    except Exception as e:

        print(f"[ERROR] Could not load model '{weights}'")
        print(f"[ERROR] {e}")

        raise


def draw_detection_counts(frame, counts: dict) -> None:
    """Draw detection counts on frame."""

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
    """Run sign detection on video and save annotated output."""

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
        f"{video_path.stem}_detected{video_path.suffix}"
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

        # Per-frame detection counts
        detection_counts = {}

        # ── Process detections ─────────────────────────────────────
        for box in result.boxes:

            cls_id = int(box.cls[0])

            confidence = float(box.conf[0])

            # Class name from trained model
            class_name = model.names[cls_id]

            detection_counts[class_name] = (
                detection_counts.get(class_name, 0) + 1
            )

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
                label = f"{class_name} {confidence:.2f}"
            else:
                label = class_name

            # Label size
            (text_w, text_h), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )

            # Label background
            cv2.rectangle(
                annotated,
                (x1, y1 - 30),
                (x1 + text_w + 10, y1),
                (0, 255, 0),
                -1,
            )

            # Label text
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

        draw_detection_counts(annotated, detection_counts)

        # ── Save frame ─────────────────────────────────────────────
        writer.write(annotated)

        frame_idx += 1

        if frame_idx % 100 == 0:

            print(
                f"  → Frame {frame_idx}/{total} processed "
                f"(detections so far: {total_dets})"
            )

    cap.release()
    writer.release()

    print(
        f"[INFO] Done. "
        f"{frame_idx} frames processed, "
        f"{total_dets} detections."
    )

    print(f"[INFO] Output saved to: {out_path}")


def print_detection_summary(model: YOLO, video_path: Path) -> None:
    """Print total detection summary."""

    print(f"\n[INFO] Generating summary for: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    class_counts = {}

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

            class_name = model.names[cls_id]

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

    cap.release()

    print("\n  Detection Summary:")

    if not class_counts:
        print("    No signs detected.")
        return

    for cls, count in sorted(class_counts.items()):

        print(f"    {cls:<20} {count:>6}")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
def main():

    global CONFIDENCE_THRESHOLD, MODEL_WEIGHTS

    parser = argparse.ArgumentParser(
        description="YOLO11 Sign Detection on video files."
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
        help="Print detection summary.",
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

    print(f"\n[INFO] All done. Output videos saved in: {args.output}")


if __name__ == "__main__":
    main()