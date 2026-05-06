"""
YOLO Object Detection for Video Files
Uses Ultralytics YOLO (supports YOLOv8 through YOLO11/v26 via ultralytics package)
Reads video clips from /Input folder and performs object detection.
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

# YOLO model — swap tag to whichever YOLO26 weight file you have, e.g.:
#   "yolo26n.pt"  (nano)   fastest, least accurate
#   "yolo26s.pt"  (small)
#   "yolo26m.pt"  (medium)
#   "yolo26l.pt"  (large)
#   "yolo26x.pt"  (xlarge) slowest, most accurate
# If you don't have YOLO26 weights yet, "yolov8n.pt" will auto-download as a fallback.
MODEL_WEIGHTS = "yolo26m.pt"

CONFIDENCE_THRESHOLD = 0.40   # Minimum confidence to display a detection
IOU_THRESHOLD        = 0.45   # NMS IoU threshold
DEVICE               = ""     # "" = auto-select (GPU if available, else CPU)
SHOW_LABELS          = True
SHOW_CONF            = True
LINE_WIDTH           = 2


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def get_video_files(folder: str) -> list[Path]:
    """Return all supported video files found in *folder*."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return [
        p for p in sorted(folder_path.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_model(weights: str) -> YOLO:
    """Load YOLO model, falling back gracefully if weights are not found."""
    try:
        model = YOLO(weights)
        print(f"[INFO] Loaded model: {weights}")
        return model
    except Exception as e:
        fallback = "yolov8n.pt"
        print(f"[WARN] Could not load '{weights}': {e}")
        print(f"[INFO] Falling back to '{fallback}' (auto-download if needed)")
        return YOLO(fallback)


def process_video(model: YOLO, video_path: Path, output_folder: str) -> None:
    """Run object detection on a single video and save the annotated result."""
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
    print(f"[INFO] Resolution: {width}x{height}  |  FPS: {fps:.1f}  |  Frames: {total}")

    # Output path
    os.makedirs(output_folder, exist_ok=True)
    out_path = Path(output_folder) / f"{video_path.stem}_detected{video_path.suffix}"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx   = 0
    total_dets  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Run inference ──────────────────────────────────────────────
        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        # ── Annotate frame ─────────────────────────────────────────────
        annotated = results[0].plot(
            labels=SHOW_LABELS,
            conf=SHOW_CONF,
            line_width=LINE_WIDTH,
        )

        # ── Overlay stats ──────────────────────────────────────────────
        n_det = len(results[0].boxes)
        total_dets += n_det
        cv2.putText(
            annotated,
            f"Frame {frame_idx+1}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  → Frame {frame_idx}/{total} processed  (dets so far: {total_dets})")

    cap.release()
    writer.release()

    print(f"[INFO] Done. {frame_idx} frames processed, {total_dets} total detections.")
    print(f"[INFO] Output saved to: {out_path}")


def print_detection_summary(model: YOLO, video_path: Path) -> None:
    """Print per-class detection counts for a quick summary pass."""
    print(f"\n[INFO] Generating detection summary for: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    class_counts: dict[str, int] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        for box in results[0].boxes:
            cls_id   = int(box.cls[0])
            cls_name = model.names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    cap.release()
    if class_counts:
        print("\n  Detection summary (total across all frames):")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<20} {count:>6}")
    else:
        print("  No detections found.")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
def main():
    # Declare globals before any reference to them
    global CONFIDENCE_THRESHOLD, MODEL_WEIGHTS

    parser = argparse.ArgumentParser(
        description="YOLO26 object detection on video files."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_VIDEO,
        help=(
            f"Path to a single video file OR a folder of videos. "
            f"Defaults to '{DEFAULT_VIDEO}'. "
            f"Pass '--input folder' to process all videos in /Input."
        ),
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FOLDER,
        help=f"Output folder for annotated videos (default: {OUTPUT_FOLDER})",
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
        help="Print per-class detection summary after processing each video.",
    )
    args = parser.parse_args()

    # Override globals from CLI args
    CONFIDENCE_THRESHOLD = args.conf
    MODEL_WEIGHTS        = args.model

    model = load_model(args.model)

    # Decide: single file or whole folder?
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