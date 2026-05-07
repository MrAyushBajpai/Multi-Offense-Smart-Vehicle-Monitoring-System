"""
YOLO Vehicle Classification for Video Files
Uses Ultralytics YOLO (supports YOLOv8 through YOLO11/v26 via ultralytics package)

Detects and classifies:
- Car
- Motorcycle
- Bus
- Truck

Reads video clips from /Input folder and saves annotated output videos.
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

# ── Preview window ─────────────────────────────────────────────────────────────
PREVIEW_ENABLED     = True
PREVIEW_WINDOW_NAME = "YOLO Vehicle Classification — press Q to quit"
PREVIEW_MAX_WIDTH   = 1280   # Down-scale preview if video is wider than this
PREVIEW_WAIT_MS     = 1      # 1 ms = as fast as possible; increase to slow preview


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".mp4", ".avi", ".mov",
    ".mkv", ".wmv", ".flv", ".webm"
}


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
        print(f"[INFO] Falling back to '{fallback}'")
        return YOLO(fallback)


def resize_for_preview(frame, max_width: int):
    """Proportionally down-scale a frame so it fits within max_width."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def open_preview_window(name: str) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowTitle(name, name)


def draw_vehicle_counts(frame, counts: dict) -> None:
    """Draw vehicle classification counts on frame (top-left)."""
    y = 30
    for label, count in counts.items():
        cv2.putText(
            frame, f"{label}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2, cv2.LINE_AA,
        )
        y += 30


def process_video(
    model: YOLO,
    video_path: Path,
    output_folder: str,
    preview: bool = PREVIEW_ENABLED,
) -> None:
    """Run vehicle classification on video, show live preview, and save output."""
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
    out_path = Path(output_folder) / f"{video_path.stem}_classified{video_path.suffix}"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    if preview:
        open_preview_window(PREVIEW_WINDOW_NAME)

    frame_idx     = 0
    total_dets    = 0
    stopped_early = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Inference ──────────────────────────────────────────────
        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        annotated = frame.copy()
        vehicle_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

        # ── Process detections ─────────────────────────────────────
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            confidence   = float(box.conf[0])
            vehicle_name = VEHICLE_CLASSES[cls_id]
            vehicle_counts[vehicle_name] += 1
            total_dets += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), LINE_WIDTH)

            label = f"{vehicle_name} {confidence:.2f}" if SHOW_CONF else vehicle_name

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - 30), (x1 + text_w + 10, y1), (0, 255, 0), -1)
            cv2.putText(
                annotated, label, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
            )

        # ── Overlays ───────────────────────────────────────────────
        draw_vehicle_counts(annotated, vehicle_counts)
        cv2.putText(
            annotated,
            f"Frame {frame_idx+1}/{total}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

        # ── Write to file ──────────────────────────────────────────
        writer.write(annotated)

        # ── Live preview ───────────────────────────────────────────
        if preview:
            cv2.imshow(PREVIEW_WINDOW_NAME, resize_for_preview(annotated, PREVIEW_MAX_WIDTH))
            key = cv2.waitKey(PREVIEW_WAIT_MS) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("\n[INFO] User pressed Q — stopping early.")
                stopped_early = True
                break
            if cv2.getWindowProperty(PREVIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("\n[INFO] Preview window closed — stopping early.")
                stopped_early = True
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  → Frame {frame_idx}/{total}  (vehicle dets so far: {total_dets})")

    cap.release()
    writer.release()

    if preview:
        cv2.destroyWindow(PREVIEW_WINDOW_NAME)

    status = "stopped early" if stopped_early else "complete"
    print(f"[INFO] Done ({status}). {frame_idx} frames, {total_dets} vehicle detections.")
    print(f"[INFO] Output saved to: {out_path}")


def print_detection_summary(model: YOLO, video_path: Path) -> None:
    print(f"\n[INFO] Generating vehicle summary for: {video_path.name}")
    cap = cv2.VideoCapture(str(video_path))
    class_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            class_counts[VEHICLE_CLASSES[cls_id]] += 1

    cap.release()

    print("\n  Vehicle Detection Summary:")
    found = any(v > 0 for v in class_counts.values())
    if found:
        for cls, count in class_counts.items():
            if count > 0:
                print(f"    {cls:<15} {count:>6}")
    else:
        print("    No vehicles detected.")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────
def main():
    global CONFIDENCE_THRESHOLD, MODEL_WEIGHTS, PREVIEW_ENABLED

    parser = argparse.ArgumentParser(
        description="YOLO Vehicle Classification on video files with live preview."
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
        help="Print vehicle classification summary after processing.",
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

    print(f"\n[INFO] All done. Classified videos are in: {args.output}")


if __name__ == "__main__":
    main()