"""
YOLO Vehicle Speed Detection for Skewed/Angled Camera Footage
Uses perspective transformation to correct for camera angle,
then measures vehicle speed using virtual measurement lines
that follow the road curvature.

Detects and tracks:
- Car
- Motorcycle
- Bus
- Truck

Reads video clips from /Input folder and saves annotated output videos.

Usage:
    python speed_detection.py
    python speed_detection.py --input Input/myvideo.mp4
    python speed_detection.py --input Input/ --calibrate
    python speed_detection.py --input Input/myvideo.mp4 --real_distance 20 --fps_override 30
"""

import cv2
import os
import argparse
import numpy as np
import time
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
INPUT_FOLDER  = "Input"
OUTPUT_FOLDER = "Output"
DEFAULT_VIDEO = "Input/input.mp4"
MODEL_WEIGHTS = "yolo26m.pt"

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD        = 0.45
DEVICE               = ""
LINE_WIDTH           = 2

# ── Speed Measurement ────────────────────
# Real-world distance (meters) between the two virtual measurement lines.
# Measure on a map or use known road markings (e.g. lane-line spacing ~6 m,
# highway lane ~30 m visible section, etc.)
REAL_DISTANCE_METERS = 20.0          # adjust to your footage

# Speed unit shown on overlay
SPEED_UNIT = "km/h"                  # "km/h" or "mph"
MPH_FACTOR = 0.621371                # km/h → mph

# Pixels-per-metre scale used ONLY when perspective transform is active.
# Leave as None to auto-derive from the warp quad dimensions.
PIXELS_PER_METRE = None

# Vehicle COCO class IDs
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Tracker history
MAX_TRACK_AGE    = 60   # frames before a lost track is discarded
SPEED_HISTORY    = 10   # rolling average window (frames)
MIN_SPEED_FRAMES = 3    # minimum crossings before reporting speed

# ── Perspective / Road Geometry ──────────
# Four points in the *original* frame that form a known rectangle on the road.
# Order: top-left, top-right, bottom-right, bottom-left  (road quad)
# Set to None to use the interactive calibration tool (--calibrate flag).
#
# Example for the snowy UK motorway shot in the sample image:
#   The road quad covers roughly the right carriageway from near to far.
#   Tune these for YOUR footage.
SRC_POINTS = None   # e.g. np.float32([[530,180],[870,180],[1100,550],[300,550]])

# How tall (px) the warped bird's-eye view should be.
# Width is derived automatically to keep real-world aspect ratio.
WARP_HEIGHT = 600

# ── Measurement lines ────────────────────
# Defined as Y-fractions of the warped image (0 = top, 1 = bottom).
# Line A is always the first the vehicle crosses (closer to top in warp = far from camera).
LINE_A_FRAC = 0.25   # upper reference line
LINE_B_FRAC = 0.60   # lower reference line

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

COLOR_GREEN  = (0, 220, 60)
COLOR_YELLOW = (0, 220, 220)
COLOR_RED    = (40, 40, 220)
COLOR_BLUE   = (220, 100, 40)
COLOR_WHITE  = (240, 240, 240)
COLOR_DARK   = (20, 20, 20)
ALPHA        = 0.45   # overlay transparency


def get_video_files(folder: str) -> list:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return [p for p in sorted(folder_path.iterdir())
            if p.suffix.lower() in SUPPORTED_EXTENSIONS]


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


# ─────────────────────────────────────────
# Interactive calibration
# ─────────────────────────────────────────
_calib_points = []
_calib_frame  = None

def _mouse_cb(event, x, y, flags, param):
    global _calib_points
    if event == cv2.EVENT_LBUTTONDOWN and len(_calib_points) < 4:
        _calib_points.append((x, y))
        print(f"  Point {len(_calib_points)}: ({x}, {y})")


def interactive_calibrate(video_path: str):
    """
    Lets user click 4 road corners on first frame.
    Order: TL, TR, BR, BL.

    Then asks for real-world distance between Line A and Line B.
    Returns:
        (points, distance_m)
    """
    global _calib_points, _calib_frame

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame from {video_path}")

    _calib_frame = frame.copy()
    _calib_points = []

    win = "Calibrate - click 4 road corners"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win,
                     min(frame.shape[1], 1280),
                     min(frame.shape[0], 720))

    cv2.imshow(win, _calib_frame)
    cv2.waitKey(1)

    cv2.setMouseCallback(win, _mouse_cb)

    print("\n[CALIBRATE]")
    print("Click 4 corners of rectangular road section.")
    print("Order: TL -> TR -> BR -> BL")
    print("Press ESC to abort.\n")

    while True:
        display = _calib_frame.copy()

        for i, pt in enumerate(_calib_points):
            cv2.circle(display, pt, 8, (0, 255, 0), -1)

            cv2.putText(display,
                        str(i + 1),
                        (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

        if len(_calib_points) == 4:
            pts = np.array(_calib_points, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        cv2.imshow(win, display)

        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            cv2.destroyWindow(win)
            raise RuntimeError("Calibration aborted.")

        if len(_calib_points) == 4:
            cv2.destroyWindow(win)
            break

    pts = np.float32(_calib_points)

    print(f"\n[CALIBRATE] Points saved:")
    print(pts.tolist())

    # Ask for distance
    while True:
        try:
            distance_m = float(
                input("\nEnter real-world distance in metres between Line A and Line B: ")
            )

            if distance_m <= 0:
                print("Distance must be > 0")
                continue

            break

        except ValueError:
            print("Invalid number. Try again.")

    print(f"[CALIBRATE] Distance set to {distance_m} metres")

    return pts, distance_m


# ─────────────────────────────────────────
# Perspective transform
# ─────────────────────────────────────────

def build_perspective_transform(src_pts: np.ndarray,
                                 real_distance_m: float,
                                 warp_height: int):
    """
    Build M (frame→warp) and Minv (warp→frame) from 4 road corners.

    src_pts order: TL, TR, BR, BL  in the original frame.
    The real-world width of the road quad is estimated from the
    real_distance_m (which spans LINE_A to LINE_B vertically).
    Returns (M, Minv, warp_w, warp_h, pixels_per_metre).
    """
    tl, tr, br, bl = src_pts

    # Destination width = average of top & bottom edge widths in source,
    # scaled by the aspect ratio of real-world distance vs width.
    top_w   = np.linalg.norm(tr - tl)
    bot_w   = np.linalg.norm(br - bl)
    src_w   = (top_w + bot_w) / 2.0

    top_h   = np.linalg.norm(bl - tl)
    bot_h   = np.linalg.norm(br - tr)
    src_h   = (top_h + bot_h) / 2.0

    aspect  = src_w / max(src_h, 1)
    warp_w  = max(int(warp_height * aspect), 100)

    dst_pts = np.float32([
        [0,        0         ],
        [warp_w,   0         ],
        [warp_w,   warp_height],
        [0,        warp_height],
    ])

    M    = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    # pixels per metre: the vertical span covers real_distance_m
    line_span_frac  = LINE_B_FRAC - LINE_A_FRAC
    ppm             = (warp_height * line_span_frac) / real_distance_m

    return M, Minv, warp_w, warp_height, ppm


# ─────────────────────────────────────────
# Draw curved measurement lines on frame
# ─────────────────────────────────────────

def warp_line_to_frame(y_frac: float,
                       warp_w: int,
                       warp_h: int,
                       Minv: np.ndarray,
                       num_pts: int = 40) -> np.ndarray:
    """
    Project a horizontal line in warped space back to the original frame.
    Returns array of (x,y) integer points suitable for cv2.polylines.
    """
    y_warp  = y_frac * warp_h
    xs      = np.linspace(0, warp_w, num_pts)
    pts_w   = np.float32([[x, y_warp] for x in xs]).reshape(-1, 1, 2)
    pts_f   = cv2.perspectiveTransform(pts_w, Minv)
    return pts_f.reshape(-1, 2).astype(np.int32)


# ─────────────────────────────────────────
# Speed tracker
# ─────────────────────────────────────────

class SpeedTracker:
    """
    Tracks per-vehicle crossing times across line A and line B
    to estimate speed.
    """

    def __init__(self, fps: float, ppm: float, real_dist_m: float):
        self.fps        = fps
        self.ppm        = ppm               # pixels per metre (in warp space)
        self.real_dist  = real_dist_m       # metres between lines

        # track_id → state dict
        self._state: dict = {}
        # track_id → speed history (km/h)
        self._speeds: dict = defaultdict(lambda: deque(maxlen=SPEED_HISTORY))
        # track_id → last seen frame
        self._last_seen: dict = {}

    def _init_track(self, tid):
        self._state[tid] = {
            "crossed_a": False,
            "frame_a":   None,
            "crossed_b": False,
            "frame_b":   None,
            "speed_kmh": None,
        }

    def update(self, tid: int, y_warp: float, warp_h: int, frame_idx: int):
        """
        Feed the warp-space Y-coordinate of vehicle centre for this frame.
        Returns current speed estimate (km/h) or None.
        """
        if tid not in self._state:
            self._init_track(tid)

        self._last_seen[tid] = frame_idx
        s = self._state[tid]

        line_a_y = LINE_A_FRAC * warp_h
        line_b_y = LINE_B_FRAC * warp_h

        # Vehicles travel downward in warp (near → far is top→bottom)
        if not s["crossed_a"] and y_warp >= line_a_y:
            s["crossed_a"] = True
            s["frame_a"]   = frame_idx

        if s["crossed_a"] and not s["crossed_b"] and y_warp >= line_b_y:
            s["crossed_b"] = True
            s["frame_b"]   = frame_idx

            dt_frames = s["frame_b"] - s["frame_a"]
            if dt_frames > 0:
                dt_sec    = dt_frames / self.fps
                speed_mps = self.real_dist / dt_sec
                speed_kmh = speed_mps * 3.6
                s["speed_kmh"] = speed_kmh
                self._speeds[tid].append(speed_kmh)

            # Reset so vehicle can be re-timed if it lingers
            s["crossed_a"] = False
            s["crossed_b"] = False

        # Smooth speed
        hist = self._speeds[tid]
        if len(hist) >= MIN_SPEED_FRAMES:
            return float(np.mean(list(hist)))
        elif s["speed_kmh"] is not None:
            return s["speed_kmh"]
        return None

    def purge_old(self, current_frame: int):
        dead = [tid for tid, f in self._last_seen.items()
                if current_frame - f > MAX_TRACK_AGE]
        for tid in dead:
            self._state.pop(tid, None)
            self._speeds.pop(tid, None)
            self._last_seen.pop(tid, None)


# ─────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────

def draw_curved_line(frame, pts: np.ndarray, color, thickness=2, label=""):
    overlay = frame.copy()
    cv2.polylines(overlay, [pts], False, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    if label:
        mid = pts[len(pts)//2]
        cv2.putText(frame, label, (int(mid[0])+8, int(mid[1])-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_speed_box(frame, x1, y1, label: str, speed_str: str, color):
    """Draw vehicle bounding-box label with speed."""
    # Background pill
    text1 = label
    text2 = speed_str

    (w1, h1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    (w2, h2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)

    bw  = max(w1, w2) + 14
    bh  = h1 + h2 + 18
    rx1 = x1
    ry1 = max(0, y1 - bh)

    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx1+bw, y1), COLOR_DARK, -1)
    cv2.addWeighted(overlay, ALPHA+0.35, frame, 1-(ALPHA+0.35), 0, frame)

    cv2.rectangle(frame, (rx1, ry1), (rx1+bw, y1), color, 1)

    cv2.putText(frame, text1, (rx1+7, ry1+h1+4),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, text2, (rx1+7, ry1+h1+h2+12),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_hud(frame, counts: dict, frame_idx: int, total: int, fps: float):
    h, w = frame.shape[:2]
    hud  = frame.copy()

    # Semi-transparent side panel
    cv2.rectangle(hud, (0, 0), (220, 30 + 28*len(counts) + 40), COLOR_DARK, -1)
    cv2.addWeighted(hud, ALPHA, frame, 1-ALPHA, 0, frame)

    cv2.putText(frame, "VEHICLE COUNTS", (10, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_YELLOW, 1, cv2.LINE_AA)

    for i, (name, cnt) in enumerate(counts.items()):
        y = 22 + 28*(i+1)
        cv2.putText(frame, f"{name:<12} {cnt:>4}",
                    (10, y), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                    COLOR_GREEN, 1, cv2.LINE_AA)

    # Frame counter bottom-left
    cv2.putText(frame, f"Frame {frame_idx+1}/{total}",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                COLOR_WHITE, 1, cv2.LINE_AA)


# ─────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────

def process_video(model: YOLO,
                  video_path: Path,
                  output_folder: str,
                  src_pts: np.ndarray,
                  real_distance: float,
                  fps_override: float | None = None) -> None:

    print(f"\n{'='*64}")
    print(f"[INFO] Processing: {video_path.name}")
    print(f"{'='*64}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return

    fps    = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] {width}x{height}  FPS:{fps:.1f}  Frames:{total}")

    # ── Perspective setup ──────────────────────────────────────────
    use_warp = src_pts is not None
    if use_warp:
        M, Minv, warp_w, warp_h, ppm = build_perspective_transform(
            src_pts, real_distance, WARP_HEIGHT)
        if PIXELS_PER_METRE:
            ppm = PIXELS_PER_METRE
        print(f"[INFO] Perspective transform: {warp_w}x{warp_h}px  "
              f"PPM={ppm:.2f}  real_dist={real_distance}m")

        # Pre-compute curved line points for overlay
        line_a_pts = warp_line_to_frame(LINE_A_FRAC, warp_w, warp_h, Minv)
        line_b_pts = warp_line_to_frame(LINE_B_FRAC, warp_w, warp_h, Minv)

        tracker = SpeedTracker(fps, ppm, real_distance)
    else:
        # Fallback: pixel-based lines across the frame
        ppm     = None
        warp_w  = width
        warp_h  = height
        line_a_y = int(height * LINE_A_FRAC)
        line_b_y = int(height * LINE_B_FRAC)
        line_a_pts = np.array([[0, line_a_y], [width, line_a_y]], np.int32)
        line_b_pts = np.array([[0, line_b_y], [width, line_b_y]], np.int32)
        tracker = SpeedTracker(fps, 1.0, real_distance)
        print("[WARN] No perspective points – speed will be in arbitrary units.")

    # ── Output writer ─────────────────────────────────────────────
    os.makedirs(output_folder, exist_ok=True)
    out_path = Path(output_folder) / f"{video_path.stem}_speed{video_path.suffix}"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx  = 0
    total_dets = 0
    vehicle_counts = {v: 0 for v in VEHICLE_CLASSES.values()}
    # Persist per-track speed between frames for smooth display
    speed_cache: dict = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLO inference with tracking ──────────────────────────
        results = model.track(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            persist=True,
            verbose=False,
        )

        result  = results[0]
        tracker.purge_old(frame_idx)

        # ── Warp frame for speed measurement ─────────────────────
        if use_warp:
            warp_frame = cv2.warpPerspective(frame, M, (warp_w, warp_h))
        else:
            warp_frame = frame   # unused but keeps logic uniform

        # ── Per-frame counts (reset each frame) ──────────────────
        frame_counts = {v: 0 for v in VEHICLE_CLASSES.values()}

        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            conf         = float(box.conf[0])
            vname        = VEHICLE_CLASSES[cls_id]
            frame_counts[vname] += 1
            total_dets   += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy          = (x1+x2)//2, (y1+y2)//2  # centre in frame

            # Track ID (may be None if tracker hasn't assigned yet)
            tid = int(box.id[0]) if box.id is not None else None

            # ── Map centre to warp space ──────────────────────────
            if use_warp and tid is not None:
                pt_src  = np.float32([[[cx, cy]]])
                pt_warp = cv2.perspectiveTransform(pt_src, M)[0][0]
                y_warp  = float(pt_warp[1])
                spd     = tracker.update(tid, y_warp, warp_h, frame_idx)
                if spd is not None:
                    speed_cache[tid] = spd
            elif tid is not None:
                # Fallback: use raw frame Y
                spd = tracker.update(tid, cy, height, frame_idx)
                if spd is not None:
                    speed_cache[tid] = spd

            # Build speed label
            if tid is not None and tid in speed_cache:
                raw_spd = speed_cache[tid]
                if SPEED_UNIT == "mph":
                    disp_spd = raw_spd * MPH_FACTOR
                else:
                    disp_spd = raw_spd
                spd_label = f"{disp_spd:.0f} {SPEED_UNIT}"
                box_color = (
                    COLOR_RED    if disp_spd > 100 else
                    COLOR_YELLOW if disp_spd > 60  else
                    COLOR_GREEN
                )
            else:
                spd_label = "-- " + SPEED_UNIT
                box_color = (160, 160, 160)

            # ── Draw bounding box ─────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, LINE_WIDTH)
            draw_speed_box(frame, x1, y1, vname, spd_label, box_color)

        # Accumulate total counts (max ever seen per class)
        for k in vehicle_counts:
            if frame_counts[k] > 0:
                vehicle_counts[k] += frame_counts[k]

        # ── Draw measurement lines ────────────────────────────────
        draw_curved_line(frame, line_a_pts, COLOR_BLUE,   2, "Line A")
        draw_curved_line(frame, line_b_pts, COLOR_YELLOW, 2, "Line B")

        # ── HUD overlay ───────────────────────────────────────────
        draw_hud(frame, frame_counts, frame_idx, total, fps)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  → {frame_idx}/{total} frames  |  dets so far: {total_dets}")

    cap.release()
    writer.release()

    print(f"\n[INFO] Done. {frame_idx} frames, {total_dets} detections.")
    print(f"[INFO] Output: {out_path}")

    # Summary
    print("\n  Vehicle Detection Summary:")
    for cls, cnt in vehicle_counts.items():
        if cnt > 0:
            print(f"    {cls:<15} {cnt:>6}")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main():
    global CONFIDENCE_THRESHOLD
    global MODEL_WEIGHTS
    global REAL_DISTANCE_METERS
    global SPEED_UNIT

    parser = argparse.ArgumentParser(
        description="YOLO Vehicle Speed Detection (perspective-corrected)."
    )
    parser.add_argument("--input",         default=DEFAULT_VIDEO,
                        help=f"Video file or folder (default: {DEFAULT_VIDEO})")
    parser.add_argument("--output",        default=OUTPUT_FOLDER,
                        help=f"Output folder (default: {OUTPUT_FOLDER})")
    parser.add_argument("--model",         default=MODEL_WEIGHTS)
    parser.add_argument("--conf",          type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--real_distance", type=float, default=REAL_DISTANCE_METERS,
                        help="Real-world distance in metres between Line A and Line B")
    parser.add_argument("--fps_override",  type=float, default=None,
                        help="Force a specific FPS (useful for VFR videos)")
    parser.add_argument("--calibrate",     action="store_true",
                        help="Launch interactive calibration tool to pick road corners")
    parser.add_argument("--unit",          choices=["km/h","mph"], default=SPEED_UNIT)

    args = parser.parse_args()

    CONFIDENCE_THRESHOLD  = args.conf
    MODEL_WEIGHTS         = args.model
    REAL_DISTANCE_METERS  = args.real_distance
    SPEED_UNIT            = args.unit

    # Collect videos
    input_path = Path(args.input)
    if input_path.is_dir():
        videos = get_video_files(str(input_path))
        if not videos:
            print(f"[WARN] No supported video files in {input_path}")
            return
    else:
        if not input_path.exists():
            print(f"[ERROR] File not found: {input_path}")
            return
        videos = [input_path]

    # Calibration: run once on first video, reuse for all
    src_pts = SRC_POINTS
    if args.calibrate or src_pts is None:
        print("[INFO] Starting calibration on:", videos[0])
        try:
            src_pts, REAL_DISTANCE_METERS = interactive_calibrate(str(videos[0]))
        except RuntimeError as e:
            print(f"[WARN] {e}")
            print("[WARN] Running WITHOUT perspective correction.")
            src_pts = None

    model = load_model(args.model)

    for video in videos:
        process_video(
            model, video, args.output,
            src_pts, REAL_DISTANCE_METERS, args.fps_override
        )

    print(f"\n[INFO] All done. Results in: {args.output}")


if __name__ == "__main__":
    main()