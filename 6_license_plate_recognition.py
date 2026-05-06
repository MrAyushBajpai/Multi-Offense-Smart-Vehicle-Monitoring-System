"""
License Plate Recognition for Video Files
==========================================
Uses:
  - Ultralytics YOLO  →  vehicle / license-plate detection
  - EasyOCR           →  OCR on cropped plate regions

Detects vehicles, localises their licence plates, and reads the plate text.
Reads video clips from /Input and writes annotated output to /Output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL OPTIONS  (set MODEL_MODE below)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Option A — morsetechlab LP model  ★ RECOMMENDED
  Repo    : https://huggingface.co/morsetechlab/license-plate-finetune
  Pick ONE of the following (place next to this script):

  Filename                          Size   Speed / Accuracy
  ──────────────────────────────────────────────────────────
  license-plate-finetune-v1n.pt     5.47 MB  ★ fastest, good accuracy
  license-plate-finetune-v1s.pt    19.2  MB    better accuracy
  license-plate-finetune-v1m.pt    40.5  MB    great balance
  license-plate-finetune-v1x.pt   114   MB    highest accuracy, slowest

  Direct download URLs (wget / browser):
  https://huggingface.co/morsetechlab/license-plate-finetune/resolve/main/license-plate-finetune-v1n.pt
  https://huggingface.co/morsetechlab/license-plate-finetune/resolve/main/license-plate-finetune-v1s.pt
  https://huggingface.co/morsetechlab/license-plate-finetune/resolve/main/license-plate-finetune-v1m.pt
  https://huggingface.co/morsetechlab/license-plate-finetune/resolve/main/license-plate-finetune-v1x.pt

Option B — Generic COCO YOLOv8n (auto-downloads on first run, no manual step)
  Weights : yolov8n.pt
  Detects vehicles (car/bus/truck/motorcycle) from COCO classes;
  the script then runs EasyOCR on the full vehicle crop.
  Accuracy is lower than Option A but needs zero setup.

Set MODEL_MODE = "lp"   for Option A
Set MODEL_MODE = "coco" for Option B
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Install dependencies:
    pip install ultralytics easyocr opencv-python
"""

import cv2
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict

import easyocr
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
INPUT_FOLDER  = "Input"
OUTPUT_FOLDER = "Output"
DEFAULT_VIDEO = "Input/LPR.mp4"

# ── Model selection ───────────────────────────────────────────
# "lp"   → dedicated licence-plate YOLO (Option A, recommended)
# "coco" → standard YOLOv8n on full vehicle crops  (Option B)
MODEL_MODE = "lp"

LP_WEIGHTS   = "license.pt"   # Option A — rename your downloaded model to license.pt
COCO_WEIGHTS = "yolov8n.pt"       # Option B weights (auto-downloaded)

# ── Detection thresholds ──────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD        = 0.45
DEVICE               = "cuda:0"    # GPU — change to "cpu" if no NVIDIA GPU

# ── OCR ──────────────────────────────────────────────────────
OCR_LANGUAGES   = ["en"]           # add e.g. "hi", "de" for other scripts
# Run OCR on every N-th frame to save time (1 = every frame)
OCR_FRAME_SKIP  = 3

# ── Visuals ───────────────────────────────────────────────────
LINE_WIDTH       = 2
PLATE_BOX_COLOR  = (0, 200, 255)   # orange-ish for plates
VEHICLE_BOX_COLOR= (0, 255, 0)

# ── COCO vehicle class IDs (used only in "coco" mode) ─────────
COCO_VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# ─────────────────────────────────────────────────────────────
# OCR helpers
# ─────────────────────────────────────────────────────────────
def build_reader() -> easyocr.Reader:
    """Initialise EasyOCR reader (downloads models on first call)."""
    print(f"[INFO] Initialising EasyOCR for languages: {OCR_LANGUAGES}")
    return easyocr.Reader(OCR_LANGUAGES, gpu=True)


def clean_plate_text(raw: str) -> str:
    """Strip characters that are never on a licence plate."""
    return re.sub(r"[^A-Z0-9\-]", "", raw.upper()).strip()


def ocr_crop(reader: easyocr.Reader, crop_bgr) -> str:
    """Run EasyOCR on a BGR image crop and return the best text."""
    if crop_bgr is None or crop_bgr.size == 0:
        return ""

    # Upscale small crops for better OCR
    h, w = crop_bgr.shape[:2]
    if w < 100:
        scale = 100 / w
        crop_bgr = cv2.resize(
            crop_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # Light pre-processing
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    results = reader.readtext(gray, detail=1)

    if not results:
        return ""

    # Pick result with highest confidence
    best = max(results, key=lambda r: r[2])
    return clean_plate_text(best[1])


# ─────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────
def load_model(weights: str, fallback: str) -> YOLO:
    """Load YOLO model with automatic fallback."""
    try:
        model = YOLO(weights)
        print(f"[INFO] Loaded model: {weights}")
        return model
    except Exception as e:
        print(f"[WARN] Could not load '{weights}': {e}")
        print(f"[INFO] Falling back to '{fallback}'")
        return YOLO(fallback)


# ─────────────────────────────────────────────────────────────
# Video helpers
# ─────────────────────────────────────────────────────────────
def get_video_files(folder: str) -> list:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return [
        p for p in sorted(folder_path.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def draw_overlay(frame, label: str, x1: int, y1: int, x2: int, y2: int,
                 box_color, text_color=(0, 0, 0)):
    """Draw a bounding box + label background on *frame* in-place."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, LINE_WIDTH)

    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - 26), (x1 + tw + 10, y1), box_color, -1)
        cv2.putText(
            frame, label, (x1 + 5, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA,
        )


def draw_plate_log(frame, plates: list, frame_h: int):
    """Draw a small log of recently seen plates in the bottom-left."""
    unique = list(dict.fromkeys(p for p in plates if p))[-6:]  # last 6 unique
    y = frame_h - 20 - (len(unique) - 1) * 26
    for plate in unique:
        cv2.putText(
            frame, f"[{plate}]",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 200, 255), 2, cv2.LINE_AA,
        )
        y += 26


# ─────────────────────────────────────────────────────────────
# Core processing — LP mode (Option A)
# ─────────────────────────────────────────────────────────────
def process_video_lp_mode(
    lp_model: YOLO,
    reader: easyocr.Reader,
    video_path: Path,
    output_folder: str,
) -> dict:
    """
    Dedicated LP-detection model:
    Each detection IS a licence plate → crop & OCR directly.
    """
    print(f"\n{'='*60}")
    print(f"[INFO] Processing (LP mode): {video_path.name}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return {}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}x{height}  FPS:{fps:.1f}  Frames:{total}")

    os.makedirs(output_folder, exist_ok=True)
    out_path = Path(output_folder) / f"{video_path.stem}_lpr{video_path.suffix}"
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    seen_plates: list = []
    plate_log: dict   = defaultdict(int)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()
        results   = lp_model(
            frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
            device=DEVICE, verbose=False,
        )

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            plate_text = ""
            if frame_idx % OCR_FRAME_SKIP == 0:
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                plate_text = ocr_crop(reader, crop)

            label = f"{plate_text}  {conf:.2f}" if plate_text else f"Plate {conf:.2f}"

            draw_overlay(
                annotated, label, x1, y1, x2, y2,
                PLATE_BOX_COLOR, text_color=(0, 0, 0),
            )

            if plate_text:
                seen_plates.append(plate_text)
                plate_log[plate_text] += 1

        # HUD
        cv2.putText(
            annotated, f"Frame {frame_idx+1}/{total}",
            (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )
        draw_plate_log(annotated, seen_plates, height)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  → {frame_idx}/{total} frames  |  plates so far: {len(plate_log)}")

    cap.release()
    writer.release()

    print(f"[INFO] Done. {frame_idx} frames, {sum(plate_log.values())} plate detections.")
    print(f"[INFO] Output: {out_path}")
    return dict(plate_log)


# ─────────────────────────────────────────────────────────────
# Core processing — COCO mode (Option B)
# ─────────────────────────────────────────────────────────────
def process_video_coco_mode(
    coco_model: YOLO,
    reader: easyocr.Reader,
    video_path: Path,
    output_folder: str,
) -> dict:
    """
    Standard COCO model:
    Detect vehicles → crop the bottom-half (where plates usually are) → OCR.
    Lower accuracy than LP mode but zero extra model download.
    """
    print(f"\n{'='*60}")
    print(f"[INFO] Processing (COCO mode): {video_path.name}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return {}

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}x{height}  FPS:{fps:.1f}  Frames:{total}")

    os.makedirs(output_folder, exist_ok=True)
    out_path = Path(output_folder) / f"{video_path.stem}_lpr{video_path.suffix}"
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    seen_plates: list = []
    plate_log: dict   = defaultdict(int)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()
        results   = coco_model(
            frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD,
            device=DEVICE, verbose=False,
        )

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_VEHICLE_CLASSES:
                continue

            conf         = float(box.conf[0])
            vehicle_name = COCO_VEHICLE_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw vehicle box
            draw_overlay(
                annotated, f"{vehicle_name} {conf:.2f}",
                x1, y1, x2, y2, VEHICLE_BOX_COLOR,
            )

            plate_text = ""
            if frame_idx % OCR_FRAME_SKIP == 0:
                # Crop the lower ~40 % of the vehicle bbox (plate area)
                plate_y1  = y1 + int((y2 - y1) * 0.60)
                plate_crop = frame[plate_y1:y2, x1:x2]
                plate_text = ocr_crop(reader, plate_crop)

                if plate_text:
                    # Draw a thin box for the guessed plate region
                    draw_overlay(
                        annotated, plate_text,
                        x1, plate_y1, x2, y2,
                        PLATE_BOX_COLOR, text_color=(0, 0, 0),
                    )
                    seen_plates.append(plate_text)
                    plate_log[plate_text] += 1

        # HUD
        cv2.putText(
            annotated, f"Frame {frame_idx+1}/{total}",
            (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200, 200, 200), 1, cv2.LINE_AA,
        )
        draw_plate_log(annotated, seen_plates, height)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  → {frame_idx}/{total} frames  |  plates so far: {len(plate_log)}")

    cap.release()
    writer.release()

    print(f"[INFO] Done. {frame_idx} frames, {sum(plate_log.values())} plate detections.")
    print(f"[INFO] Output: {out_path}")
    return dict(plate_log)


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
def print_summary(plate_log: dict, video_name: str):
    print(f"\n{'─'*50}")
    print(f"  Licence Plate Summary — {video_name}")
    print(f"{'─'*50}")

    if not plate_log:
        print("  No plates read.")
        return

    sorted_plates = sorted(plate_log.items(), key=lambda x: -x[1])
    print(f"  {'Plate':<20}  {'Hits':>6}")
    print(f"  {'─'*20}  {'─'*6}")
    for plate, count in sorted_plates:
        print(f"  {plate:<20}  {count:>6}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    global CONFIDENCE_THRESHOLD, MODEL_MODE
    global OCR_FRAME_SKIP, OCR_LANGUAGES

    parser = argparse.ArgumentParser(
        description="Licence Plate Recognition on video files."
    )
    parser.add_argument(
        "--input", default=DEFAULT_VIDEO,
        help=f"Video file OR folder of videos. Default: '{DEFAULT_VIDEO}'",
    )
    parser.add_argument(
        "--output", default=OUTPUT_FOLDER,
        help=f"Output folder (default: {OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--mode", default=MODEL_MODE, choices=["lp", "coco"],
        help=(
            "Detection mode: "
            "'lp' = dedicated LP model (recommended, needs yolov8n-lp.pt), "
            "'coco' = generic YOLO vehicle detection (no extra download)"
        ),
    )
    parser.add_argument(
        "--conf", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print plate-count summary after each video.",
    )
    parser.add_argument(
        "--ocr-skip", type=int, default=OCR_FRAME_SKIP,
        help=f"Run OCR every N frames (default: {OCR_FRAME_SKIP}). Higher = faster.",
    )
    parser.add_argument(
        "--lang", nargs="+", default=OCR_LANGUAGES,
        help="EasyOCR language codes, e.g. --lang en hi (default: en)",
    )

    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf
    MODEL_MODE           = args.mode
    OCR_FRAME_SKIP  = args.ocr_skip
    OCR_LANGUAGES   = args.lang

    # ── Load models ──────────────────────────────────────────
    if MODEL_MODE == "lp":
        model = load_model(LP_WEIGHTS, COCO_WEIGHTS)
    else:
        model = load_model(COCO_WEIGHTS, "yolov8n.pt")

    reader = build_reader()

    # ── Resolve input ────────────────────────────────────────
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

    # ── Process ──────────────────────────────────────────────
    process_fn = (
        process_video_lp_mode if MODEL_MODE == "lp"
        else process_video_coco_mode
    )

    for video in videos:
        plate_log = process_fn(model, reader, video, args.output)
        if args.summary or True:          # always show summary
            print_summary(plate_log, video.name)

    print(f"[INFO] All done. Annotated videos saved to: {args.output}")


if __name__ == "__main__":
    main()