"""
Cricket Ball Tracker — Detection and Trajectory Rendering Engine
================================================================

Class-based wrapper around the two-pass pipeline for use as an
importable module (FastAPI, scripts, notebooks).

    tracker = CricketBallTracker(config)
    result  = tracker.process_video(video_path, out_dir, progress_cb)

Pass 1 — YOLO + ByteTrack + momentum-gated tracking + HSV recovery
Pass 2 — Bounce-first per-segment RANSAC + two-parabola fitting +
          post-bounce extension + perspective-scaled rendering
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# =====================================================================
#  CONFIGURATION (dataclass — every constant is overridable)
# =====================================================================

@dataclass
class TrackerConfig:

    # -- Paths --
    model_path: str = "cricket_yolov8/best.onnx"

    # -- YOLO --
    imgsz: int = 1280
    conf: float = 0.2
    iou: float = 0.7
    device: Optional[str] = None
    classes: list[int] = field(default_factory=lambda: [0])

    # -- ByteTrack --
    track_buffer: int = 90
    track_high_thresh: float = 0.35
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.45
    match_thresh: float = 0.80

    # -- Single-ball gating --
    gate_radius_px: int = 80
    gate_grow_per_miss: int = 15
    max_misses: int = 60

    # -- Momentum --
    vel_ema_momentum: float = 0.85
    vel_min_disp_px: float = 2.0
    gate_along_ratio: float = 1.3
    gate_perp_ratio: float = 0.6
    size_ratio_max: float = 3.0
    size_ema_momentum: float = 0.8

    # -- False-positive filtering --
    min_box_area_px: int = 25

    # -- HSV recovery --
    enable_hsv_recovery: bool = True
    hsv_h_margin: int = 12
    hsv_s_margin: int = 70
    hsv_v_margin: int = 70
    hsv_min_s: int = 40
    hsv_min_v: int = 40
    hsv_roi_pad_px: int = 40
    hsv_min_area_px: int = 6
    hsv_max_area_px: int = 2000
    hsv_aspect_min: float = 0.35
    hsv_aspect_max: float = 2.8
    hsv_morph_kernel: int = 3
    hsv_morph_iters: int = 1

    # -- Trajectory fitting --
    traj_min_points: int = 5
    traj_ransac_iters: int = 60
    traj_ransac_inlier_px: float = 10.0
    traj_bounce_thresh: int = 8
    traj_angle_cutoff: int = 45
    traj_post_bounce_extend: float = 0.4
    traj_min_post_bounce: int = 3
    traj_max_frame_gap: int = 6

    # -- Rendering --
    traj_thickness_max: int = 6
    traj_thickness_min: int = 1
    traj_dot_max: int = 6
    traj_dot_min: int = 2
    traj_bounce_ring_r1: int = 14
    traj_bounce_ring_r2: int = 9

    # Colours (BGR)
    colour_start: tuple = (100, 220, 60)
    colour_mid: tuple = (40, 200, 240)
    colour_end: tuple = (30, 80, 240)
    colour_bounce: tuple = (0, 100, 255)

    # -- Output control --
    save_annotated_video: bool = True
    save_trajectory_video: bool = True
    save_trajectory_csv: bool = True


# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================

def _write_bytetrack_yaml(path: Path, cfg: TrackerConfig):
    path.write_text(
        f"tracker_type: bytetrack\n"
        f"track_high_thresh: {cfg.track_high_thresh}\n"
        f"track_low_thresh: {cfg.track_low_thresh}\n"
        f"new_track_thresh: {cfg.new_track_thresh}\n"
        f"track_buffer: {cfg.track_buffer}\n"
        f"match_thresh: {cfg.match_thresh}\n"
        f"fuse_score: true\n",
        encoding="utf-8",
    )


def _area(x1, y1, x2, y2):
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _center(x1, y1, x2, y2):
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _clip(v, lo, hi):
    return int(max(lo, min(hi, v)))

def _roi(cx, cy, half, w, h):
    x1 = _clip(cx - half, 0, w - 1); y1 = _clip(cy - half, 0, h - 1)
    x2 = _clip(cx + half, 0, w - 1); y2 = _clip(cy + half, 0, h - 1)
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def _lerp_colour(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _traj_colour(index, total, cfg: TrackerConfig):
    if total <= 1:
        return cfg.colour_mid
    t = index / (total - 1)
    if t < 0.5:
        return _lerp_colour(cfg.colour_start, cfg.colour_mid, t * 2.0)
    return _lerp_colour(cfg.colour_mid, cfg.colour_end, (t - 0.5) * 2.0)


# =====================================================================
#  ELLIPTICAL GATING
# =====================================================================

def _elliptical_gate_dist(cand_xy, pred_xy, vel, gate, cfg: TrackerConfig):
    dx = cand_xy[0] - pred_xy[0]
    dy = cand_xy[1] - pred_xy[1]
    speed = math.hypot(vel[0], vel[1])
    if speed < 1.0:
        return math.hypot(dx, dy) / gate if gate > 0 else float("inf")
    ux, uy = vel[0] / speed, vel[1] / speed
    d_along = dx * ux + dy * uy
    d_perp = -dx * uy + dy * ux
    a = max(1.0, gate * cfg.gate_along_ratio)
    b = max(1.0, gate * cfg.gate_perp_ratio)
    return math.sqrt((d_along / a) ** 2 + (d_perp / b) ** 2)


# =====================================================================
#  HSV COLOUR RECOVERY
# =====================================================================

def _learn_hsv(frame, xyxy, cfg: TrackerConfig):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = _clip(x1, 0, w - 1); x2 = _clip(x2, 0, w - 1)
    y1 = _clip(y1, 0, h - 1); y2 = _clip(y2, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int32)
    S = hsv[:, :, 1].astype(np.int32)
    V = hsv[:, :, 2].astype(np.int32)
    good = (S >= cfg.hsv_min_s) & (V >= cfg.hsv_min_v)
    if good.sum() < 20:
        good = np.ones_like(H, dtype=bool)
    h_med = int(np.median(H[good]))
    s_med = int(np.median(S[good]))
    v_med = int(np.median(V[good]))
    s_lo = _clip(s_med - cfg.hsv_s_margin, 0, 255)
    s_hi = _clip(s_med + cfg.hsv_s_margin, 0, 255)
    v_lo = _clip(v_med - cfg.hsv_v_margin, 0, 255)
    v_hi = _clip(v_med + cfg.hsv_v_margin, 0, 255)
    h_lo = (h_med - cfg.hsv_h_margin) % 180
    h_hi = (h_med + cfg.hsv_h_margin) % 180
    if h_lo <= h_hi:
        ranges = [((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))]
    else:
        ranges = [
            ((0, s_lo, v_lo), (h_hi, s_hi, v_hi)),
            ((h_lo, s_lo, v_lo), (179, s_hi, v_hi)),
        ]
    return {"hue_ranges": ranges}


def _hsv_blob(frame, roi_xyxy, model, pred_xy, cfg: TrackerConfig):
    if model is None:
        return None
    H, W = frame.shape[:2]
    rx1, ry1, rx2, ry2 = map(int, roi_xyxy)
    rx1 = _clip(rx1, 0, W - 1); rx2 = _clip(rx2, 0, W - 1)
    ry1 = _clip(ry1, 0, H - 1); ry2 = _clip(ry2, 0, H - 1)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in model["hue_ranges"]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.bitwise_and(mask, cv2.inRange(hsv[:, :, 1], cfg.hsv_min_s, 255))
    mask = cv2.bitwise_and(mask, cv2.inRange(hsv[:, :, 2], cfg.hsv_min_v, 255))
    k = cfg.hsv_morph_kernel
    if k >= 3:
        kern = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=cfg.hsv_morph_iters)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=cfg.hsv_morph_iters)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None; best_score = float("inf")
    for c in contours:
        a = cv2.contourArea(c)
        if a < cfg.hsv_min_area_px or a > cfg.hsv_max_area_px:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 0 or bh <= 0:
            continue
        asp = bw / float(bh)
        if asp < cfg.hsv_aspect_min or asp > cfg.hsv_aspect_max:
            continue
        cx = x + 0.5 * bw; cy = y + 0.5 * bh
        if pred_xy is not None:
            score = math.hypot(cx - (pred_xy[0] - rx1), cy - (pred_xy[1] - ry1))
        else:
            score = a
        if score < best_score:
            best_score = score
            best = (x, y, x + bw, y + bh)
    if best is None:
        return None
    bx1, by1, bx2, by2 = best
    return (float(bx1 + rx1), float(by1 + ry1), float(bx2 + rx1), float(by2 + ry1))


# =====================================================================
#  PASS 2 — TRAJECTORY FITTING
# =====================================================================

def _trim_at_gap(frames, x, y, max_gap):
    for i in range(1, len(frames)):
        if frames[i] - frames[i - 1] > max_gap:
            return frames[:i], x[:i], y[:i]
    return frames, x, y


def _detect_bounce(y_raw, thresh, min_pts):
    n = len(y_raw)
    if n < min_pts:
        return None
    dy = np.diff(y_raw)
    for i in range(2, len(dy) - 1):
        if dy[i - 1] > 0 and dy[i] < 0:
            local = y_raw[max(0, i - 2): min(n, i + 3)]
            if np.ptp(local) > thresh:
                return i
    return None


def _find_cutoff(x, y, bounce_idx, angle_cutoff):
    if bounce_idx is None:
        return len(x) - 1
    for i in range(bounce_idx + 2, len(x) - 1):
        v1 = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
        v2 = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 0.5 or m2 < 0.5:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1, 1)
        if np.degrees(np.arccos(cos_a)) > angle_cutoff:
            return i
    return len(x) - 1


def _ransac_filter(x, y, n_iters, threshold, min_pts):
    n = len(x)
    if n < 4:
        return np.ones(n, dtype=bool)
    best = np.zeros(n, dtype=bool); best_cnt = 0
    for _ in range(n_iters):
        idx = np.random.choice(n, 3, replace=False)
        if len(set(x[idx])) < 3:
            continue
        try:
            c = np.polyfit(x[idx], y[idx], 2)
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(y - np.polyval(c, x)) < threshold
        cnt = inl.sum()
        if cnt > best_cnt:
            best_cnt = cnt; best = inl
    if best_cnt < min(min_pts, n):
        return np.ones(n, dtype=bool)
    return best


def _fit_parabola(x, y):
    if len(x) < 3:
        return None
    try:
        return np.polyfit(x, y, 2)
    except np.linalg.LinAlgError:
        return None


def _fit_two_parabolas(x_raw, y_raw, bounce_idx, cutoff_idx, cfg: TrackerConfig):
    # Pre-bounce
    xp = x_raw[:bounce_idx + 1]; yp = y_raw[:bounce_idx + 1]
    mp = _ransac_filter(xp, yp, cfg.traj_ransac_iters, cfg.traj_ransac_inlier_px, cfg.traj_min_points)
    xpc = xp[mp]; ypc = yp[mp]
    cp = _fit_parabola(xpc, ypc)
    ypf = np.polyval(cp, xpc) if cp is not None else ypc.copy()

    # Post-bounce
    pe = min(cutoff_idx + 1, len(x_raw))
    xq = x_raw[bounce_idx:pe]; yq = y_raw[bounce_idx:pe]
    cq = None; xqc = xq.copy(); yqf = yq.copy()
    x_ext = np.array([]); y_ext = np.array([])

    if len(xq) >= cfg.traj_min_post_bounce:
        mq = _ransac_filter(xq, yq, cfg.traj_ransac_iters, cfg.traj_ransac_inlier_px, cfg.traj_min_points)
        xqc = xq[mq]; yqc = yq[mq]
        if len(xqc) >= cfg.traj_min_post_bounce:
            cq = _fit_parabola(xqc, yqc)
        if cq is not None:
            yqf = np.polyval(cq, xqc)
            span = xqc[-1] - xqc[0]
            if span > 1.0:
                ed = span * cfg.traj_post_bounce_extend
                xe = np.linspace(xqc[-1], xqc[-1] + ed, 20)
                ye = np.polyval(cq, xe)
                good = np.ones(len(xe), dtype=bool)
                for j in range(1, len(xe)):
                    if ye[j] > ye[j - 1] or ye[j] < 0:
                        good[j:] = False; break
                if good.any():
                    x_ext = xe[good]; y_ext = ye[good]
        else:
            xqc = xq.copy(); yqf = yq.copy()

    by = ypf[-1] if cp is not None else y_raw[bounce_idx]
    bp = (float(x_raw[bounce_idx]), float(by))
    return cp, cq, xpc, ypf, xqc, yqf, x_ext, y_ext, bp


def _interp_dense(cp, cq, xp, ypf, xq, yqf, x_ext, y_ext, density=4):
    if cp is not None and len(xp) >= 2:
        xd1 = np.linspace(xp[0], xp[-1], max(len(xp) * density, 2))
        yd1 = np.polyval(cp, xd1)
    else:
        xd1 = xp.astype(float); yd1 = ypf.astype(float)
    bi = len(xd1) - 1
    if cq is not None and len(xq) >= 2:
        xd2 = np.linspace(xq[0], xq[-1], max(len(xq) * density, 2))
        yd2 = np.polyval(cq, xd2)
    else:
        xd2 = xq.astype(float); yd2 = yqf.astype(float)
    xd = np.concatenate([xd1, xd2[1:]]); yd = np.concatenate([yd1, yd2[1:]])
    if len(x_ext) > 0:
        xd = np.concatenate([xd, x_ext]); yd = np.concatenate([yd, y_ext])
    return xd, yd, bi


def _load_and_fit(csv_path: str, cfg: TrackerConfig) -> dict:
    df = pd.read_csv(csv_path)
    mask = df["x"].notna() & df["y"].notna()
    if "status" in df.columns:
        mask &= df["status"].isin(["det", "hsv", "detected"])
    valid = df[mask].reset_index(drop=True)
    if len(valid) < cfg.traj_min_points:
        raise ValueError(f"Insufficient points: {len(valid)}")
    x = valid["x"].values.astype(float)
    y = valid["y"].values.astype(float)
    fr = valid["frame"].values.astype(int)

    fr, x, y = _trim_at_gap(fr, x, y, cfg.traj_max_frame_gap)
    if len(x) < cfg.traj_min_points:
        raise ValueError("Too few points after gap trim")

    bi = _detect_bounce(y, cfg.traj_bounce_thresh, cfg.traj_min_points)
    co = _find_cutoff(x, y, bi, cfg.traj_angle_cutoff)
    if bi is not None and bi > co:
        bi = None

    if bi is not None:
        cp, cq, xp, ypf, xq, yqf, xe, ye, bp = _fit_two_parabolas(x, y, bi, co, cfg)
        xd, yd, bdi = _interp_dense(cp, cq, xp, ypf, xq, yqf, xe, ye)
    else:
        xu = x[:co + 1]; yu = y[:co + 1]
        im = _ransac_filter(xu, yu, cfg.traj_ransac_iters, cfg.traj_ransac_inlier_px, cfg.traj_min_points)
        xc = xu[im]; yc = yu[im]
        c = _fit_parabola(xc, yc)
        if c is not None:
            xd = np.linspace(xc[0], xc[-1], max(len(xc) * 4, 2))
            yd = np.polyval(c, xd)
        else:
            xd = xc.astype(float); yd = yc.astype(float)
        bp = None; bdi = None

    pts = np.column_stack([xd, yd]).astype(np.int32)
    bpf = None
    if bdi is not None and bdi < len(pts):
        bpf = (float(xd[bdi]), float(yd[bdi]))
    elif bp is not None:
        bpf = bp

    ymin, ymax = yd.min(), yd.max()
    yr = ymax - ymin
    dep = (yd - ymin) / yr if yr > 1.0 else np.full(len(yd), 0.5)
    fru = fr[:co + 1]

    return {
        "points": pts, "bounce": bpf, "bounce_idx": bdi,
        "depth_values": dep,
        "start_frame": int(fru[0]), "end_frame": int(fru[-1]),
    }


# =====================================================================
#  RENDERING
# =====================================================================

def _depth_thick(d, lo, hi):
    return max(1, int(round(lo + (hi - lo) * d)))


def _draw_stroke(frame, pts, depths, total, cfg):
    for i in range(1, len(pts)):
        d = 0.5 * (depths[i - 1] + depths[i])
        cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]),
                 _traj_colour(i, total, cfg),
                 _depth_thick(d, cfg.traj_thickness_min, cfg.traj_thickness_max),
                 cv2.LINE_AA)


def _draw_dots(frame, pts, depths, total, cfg, step=4):
    for i in range(0, len(pts), step):
        cv2.circle(frame, tuple(pts[i]),
                   _depth_thick(depths[i], cfg.traj_dot_min, cfg.traj_dot_max),
                   _traj_colour(i, total, cfg), -1, cv2.LINE_AA)


def _draw_bounce(frame, bp, depth, cfg):
    if bp is None:
        return
    cx, cy = int(bp[0]), int(bp[1])
    r1 = _depth_thick(depth, cfg.traj_bounce_ring_r1 // 2, cfg.traj_bounce_ring_r1)
    r2 = _depth_thick(depth, cfg.traj_bounce_ring_r2 // 2, cfg.traj_bounce_ring_r2)
    cv2.circle(frame, (cx, cy), r1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r2, cfg.colour_bounce, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, cfg.colour_bounce, -1, cv2.LINE_AA)


def _overlay(frame, traj, cfg):
    pts = traj["points"]; dep = traj["depth_values"]
    bp = traj["bounce"]; bi = traj["bounce_idx"]
    n = len(pts)
    if n < 2:
        return frame
    _draw_stroke(frame, pts, dep, n, cfg)
    _draw_dots(frame, pts, dep, n, cfg)
    bd = dep[bi] if bi is not None and bi < len(dep) else 0.5
    _draw_bounce(frame, bp, bd, cfg)
    return frame


# =====================================================================
#  MAIN CLASS
# =====================================================================

class CricketBallTracker:
    """End-to-end cricket ball tracking and trajectory rendering."""

    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self._model: YOLO | None = None

    # Lazy-load the YOLO model (heavy; only load once)
    def _get_model(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.cfg.model_path)
        return self._model

    def warmup(self):
        """Pre-load the YOLO model so the first request is not slow."""
        self._get_model()
        return self

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        *,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """Run the full two-pass pipeline.

        Args:
            video_path:  Path to the input video.
            output_dir:  Directory for all outputs.
            progress_callback:  Optional ``fn(current, total, message)``.

        Returns:
            dict with keys: detection_video, trajectory_video, csv,
            bounce, num_detections, fps.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _cb(cur, tot, msg):
            if progress_callback:
                progress_callback(cur, tot, msg)

        # Pass 1
        csv_path, fps = self._detection_pass(video_path, out, _cb)

        # Pass 2
        traj_video = None
        bounce = None
        if self.cfg.save_trajectory_video:
            try:
                traj_video, bounce = self._rendering_pass(
                    video_path, csv_path, out, fps, _cb)
            except ValueError as e:
                _cb(0, 0, f"Pass 2 skipped: {e}")

        return {
            "detection_video": str(out / "annotated_detection.mp4"),
            "trajectory_video": str(traj_video) if traj_video else None,
            "csv": str(csv_path),
            "bounce": bounce,
            "fps": fps,
        }

    # ------------------------------------------------------------------
    #  Pass 1 — Detection
    # ------------------------------------------------------------------

    def _detection_pass(self, video_path, out_dir, cb):
        cfg = self.cfg
        tracker_yaml = out_dir / "bytetrack_custom.yaml"
        _write_bytetrack_yaml(tracker_yaml, cfg)

        model = self._get_model()
        stream = model.track(
            source=video_path, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
            device=cfg.device, classes=cfg.classes,
            tracker=str(tracker_yaml), persist=True, stream=True,
            verbose=False, save=False,
        )

        writer = None; csv_rows = []; fps = 30.0
        active_id = None; last_pos = None
        vel = np.array([0.0, 0.0], dtype=float)
        misses = 0; hsv_model = None; avg_area = None

        # Count total frames for progress
        cap_tmp = cv2.VideoCapture(video_path)
        total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fv = cap_tmp.get(cv2.CAP_PROP_FPS)
        if fv and fv > 0:
            fps = fv
        cap_tmp.release()

        cb(0, total_frames, "Pass 1: detecting ball")

        for fi, result in enumerate(stream):
            frame = result.orig_img
            if frame is None:
                continue

            if writer is None and cfg.save_annotated_video:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(out_dir / "annotated_detection.mp4"), fourcc, fps, (w, h))

            # Candidates
            candidates = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy_arr = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
                ids = None
                if getattr(result.boxes, "id", None) is not None:
                    ids = result.boxes.id.cpu().numpy().astype(int)
                for i in range(len(xyxy_arr)):
                    x1, y1, x2, y2 = xyxy_arr[i]
                    ba = _area(x1, y1, x2, y2)
                    if cfg.min_box_area_px > 0 and ba < cfg.min_box_area_px:
                        continue
                    if avg_area is not None and avg_area > 0:
                        ratio = ba / avg_area
                        if ratio > cfg.size_ratio_max or ratio < (1.0 / cfg.size_ratio_max):
                            continue
                    candidates.append({
                        "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                        "conf": float(confs[i]),
                        "id": int(ids[i]) if ids is not None else None,
                        "area": ba,
                    })

            # Prediction
            pred = None
            if last_pos is not None:
                pred = (last_pos[0] + float(vel[0]), last_pos[1] + float(vel[1]))

            # Gating
            chosen = None
            gate = cfg.gate_radius_px + misses * cfg.gate_grow_per_miss

            if active_id is None:
                if candidates:
                    chosen = max(candidates, key=lambda d: d["conf"])
                    active_id = chosen["id"]
            else:
                same = [d for d in candidates if d["id"] == active_id]
                pool = same if same else candidates
                if pool and pred is not None:
                    scored = sorted(
                        [(cfg and _elliptical_gate_dist(
                            _center(*d["xyxy"]), pred, vel, gate, cfg), d)
                         for d in pool],
                        key=lambda t: t[0])
                    if scored[0][0] <= 1.0:
                        chosen = scored[0][1]
                elif pool:
                    chosen = max(pool, key=lambda d: d["conf"])

            # HSV recovery
            hsv_recovered = False
            if chosen is not None and cfg.enable_hsv_recovery:
                hsv_model = _learn_hsv(frame, chosen["xyxy"], cfg)
            if (chosen is None and cfg.enable_hsv_recovery
                    and hsv_model is not None and pred is not None):
                fh, fw = frame.shape[:2]
                half = int(gate + cfg.hsv_roi_pad_px)
                r = _roi(pred[0], pred[1], half, fw, fh)
                bb = _hsv_blob(frame, r, hsv_model, pred, cfg)
                if bb is not None:
                    chosen = {"xyxy": bb, "conf": 0.0, "id": active_id,
                              "area": _area(*bb)}
                    hsv_recovered = True

            # State update
            status = "miss"
            if chosen is not None:
                cx, cy = _center(*chosen["xyxy"])
                if last_pos is not None:
                    dx = cx - last_pos[0]; dy = cy - last_pos[1]
                    if math.hypot(dx, dy) >= cfg.vel_min_disp_px:
                        nv = np.array([dx, dy], dtype=float)
                        vel = cfg.vel_ema_momentum * vel + (1 - cfg.vel_ema_momentum) * nv
                last_pos = (cx, cy); misses = 0
                status = "hsv" if hsv_recovered else "det"
                if chosen["id"] is not None:
                    active_id = chosen["id"]
                ba = chosen.get("area", _area(*chosen["xyxy"]))
                avg_area = ba if avg_area is None else (
                    cfg.size_ema_momentum * avg_area + (1 - cfg.size_ema_momentum) * ba)
            else:
                misses += 1
                status = "pred" if pred is not None else "miss"
                if misses > cfg.max_misses:
                    active_id = None; last_pos = None; vel[:] = 0.0
                    hsv_model = None; avg_area = None; misses = 0
                    status = "reset"

            # Detection video
            if writer is not None:
                vis = frame.copy()
                if chosen is not None:
                    bx1, by1, bx2, by2 = map(int, chosen["xyxy"])
                    col = (0, 255, 255) if hsv_recovered else (0, 255, 0)
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), col, 2)
                writer.write(vis)

            if cfg.save_trajectory_csv:
                csv_rows.append({
                    "frame": fi, "status": status,
                    "id": active_id if active_id is not None else "",
                    "x": last_pos[0] if last_pos is not None else "",
                    "y": last_pos[1] if last_pos is not None else "",
                    "vx": float(vel[0]), "vy": float(vel[1]),
                    "gate": gate, "misses": misses,
                })

            if fi % 10 == 0:
                cb(fi, total_frames, "Pass 1: detecting ball")

        if writer is not None:
            writer.release()

        csv_path = out_dir / "trajectory.csv"
        if cfg.save_trajectory_csv and csv_rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                w.writeheader(); w.writerows(csv_rows)

        cb(total_frames, total_frames, "Pass 1: complete")
        return csv_path, fps

    # ------------------------------------------------------------------
    #  Pass 2 — Trajectory fitting + rendering
    # ------------------------------------------------------------------

    def _rendering_pass(self, video_path, csv_path, out_dir, fps, cb):
        cfg = self.cfg
        output_video = out_dir / "trajectory_output.mp4"
        traj = _load_and_fit(str(csv_path), cfg)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        cb(0, total, "Pass 2: rendering trajectory")
        fn = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fn >= traj["start_frame"]:
                frame = _overlay(frame, traj, cfg)
            writer.write(frame)
            fn += 1
            if fn % 10 == 0:
                cb(fn, total, "Pass 2: rendering trajectory")

        cap.release(); writer.release()
        cb(total, total, "Pass 2: complete")
        return output_video, traj.get("bounce")
