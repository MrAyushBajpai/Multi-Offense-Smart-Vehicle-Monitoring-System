"""
YOLO Object Detection for Video Files
Uses Ultralytics YOLO (supports YOLOv8 through YOLO11/v26 via ultralytics package)
Reads video clips from /Input folder and performs object detection.

Real-time preview: a window shows annotated frames as they are processed.
Press  Q  at any time to stop early.
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

MODEL_WEIGHTS        = "yolo26m.pt"
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD        = 0.45
DEVICE               = ""        # "" = auto-select (GPU if available, else CPU)
SHOW_LABELS          = True
SHOW_CONF            = True
LINE_WIDTH           = 2

# ── Preview window ─────────────────────────────────────────────────────────────
PREVIEW_ENABLED      = True      # Set False to disable the popup window entirely
PREVIEW_WINDOW_NAME  = "YOLO Detection — press Q to quit"
PREVIEW_MAX_WIDTH    = 1280      # Down-scale preview if video is wider than this
PREVIEW_WAIT_MS      = 1         # 1 ms keeps the window responsive; increase to slow preview


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def get_video_files(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return [
        p for p in sorted(folder_path.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def load_model(weights: str) -> YOLO:
    try:
        model = YOLO(weights)
        print(f"[INFO] Loaded model: {weights}")
        return model
    except Exception as e:
        fallback = "yolov8n.pt"
        print(f"[WARN] Could not load '{weights}': {e}")
        print(f"[INFO] Falling back to '{fallback}' (auto-download if needed)")
        return YOLO(fallback)


def resize_for_preview(frame, max_width: int):
    """Proportionally down-scale a frame so it fits within max_width."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale  = max_width / w
    new_wh = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_wh, interpolation=cv2.INTER_LINEAR)


def open_preview_window(name: str) -> None:
    """Create a named, resizable preview window."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowTitle(name, name)


def process_video(
    model: YOLO,
    video_path: Path,
    output_folder: str,
    preview: bool = PREVIEW_ENABLED,
) -> None:
    """Run object detection on a single video, show live preview, and save output."""
    print(f"\n{'='*60}")
    print(f"[INFO] Processing: {video_path.name}")
    if preview:
        print(f"[INFO] Preview window open — press  Q  to stop early")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Resolution: {width}x{height}  |  FPS: {fps:.1f}  |  Frames: {total}")

    os.makedirs(output_folder, exist_ok=True)
    out_path = Path(output_folder) / f"{video_path.stem}_detected{video_path.suffix}"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if preview:
        open_preview_window(PREVIEW_WINDOW_NAME)

    frame_idx  = 0
    total_dets = 0
    stopped_early = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Inference ──────────────────────────────────────────────────
        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        # ── Annotate ───────────────────────────────────────────────────
        annotated = results[0].plot(
            labels=SHOW_LABELS,
            conf=SHOW_CONF,
            line_width=LINE_WIDTH,
        )

        # ── Frame counter overlay ──────────────────────────────────────
        n_det = len(results[0].boxes)
        total_dets += n_det
        cv2.putText(
            annotated,
            f"Frame {frame_idx+1}/{total}  |  Dets: {n_det}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # ── Write to file ──────────────────────────────────────────────
        writer.write(annotated)

        # ── Live preview ───────────────────────────────────────────────
        if preview:
            preview_frame = resize_for_preview(annotated, PREVIEW_MAX_WIDTH)
            cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)
            key = cv2.waitKey(PREVIEW_WAIT_MS) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("\n[INFO] User pressed Q — stopping early.")
                stopped_early = True
                break

            # Also stop if the user closed the window via the X button
            if cv2.getWindowProperty(PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[INFO] Preview window closed — stopping early.")
                stopped_early = True
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  → Frame {frame_idx}/{total}  (dets so far: {total_dets})")

    cap.release()
    writer.release()

    if preview:
        cv2.destroyWindow(PREVIEW_WINDOW_NAME)

    status = "stopped early" if stopped_early else "complete"
    print(f"[INFO] Done ({status}). {frame_idx} frames, {total_dets} total detections.")
    print(f"[INFO] Output saved to: {out_path}")


def print_detection_summary(model: YOLO, video_path: Path) -> None:
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
    global CONFIDENCE_THRESHOLD, MODEL_WEIGHTS, PREVIEW_ENABLED

    parser = argparse.ArgumentParser(
        description="YOLO object detection on video files with live preview."
    )
    parser.add_argument(
        "--input", default=DEFAULT_VIDEO,
        help=f"Path to a video file or folder of videos (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--output", default=OUTPUT_FOLDER,
        help=f"Output folder for annotated videos (default: {OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--model", default=MODEL_WEIGHTS,
        help=f"YOLO weights file (default: {MODEL_WEIGHTS})",
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Disable the real-time preview window (headless / server mode)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print per-class detection summary after processing each video.",
    )
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf
    MODEL_WEIGHTS        = args.model
    PREVIEW_ENABLED      = not args.no_preview

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
        process_video(model, video, args.output, preview=PREVIEW_ENABLED)
        if args.summary:
            print_detection_summary(model, video)

    print(f"\n[INFO] All done. Annotated videos are in: {args.output}")


if __name__ == "__main__":
    main()