"""
Cricket Ball Tracker — Detection and Trajectory Rendering Engine
================================================================

Class-based wrapper around the two-pass pipeline for use as an
importable module (FastAPI, scripts, notebooks).

    tracker = CricketBallTracker(config)
    result  = tracker.process_video(video_path, out_dir, progress_cb)

Pass 1 — YOLO + ByteTrack + Euclidean-gated tracking + HSV recovery
Pass 2 — IMM+UKF speed estimation + parabolic smoothing + trajectory overlay

Based on hsv_imm_pipeline.py (merged reference implementation).
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
from scipy.linalg import cholesky
from ultralytics import YOLO


# =====================================================================
#  CONFIGURATION
# =====================================================================

@dataclass
class TrackerConfig:

    # -- Paths --
    model_path: str = "cricket_yolov8/best.onnx"

    # -- YOLO --
    imgsz: int = 1280  # Must be 1280 to ensure accuracy
    conf: float = 0.20
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
    min_box_area_px: int = 25

    # -- Velocity EMA (alpha=0.3 → 0.7*old + 0.3*new) --
    vel_ema_alpha: float = 0.3

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

    # -- Trajectory --
    traj_min_points: int = 5
    traj_bounce_thresh: int = 8
    traj_angle_cutoff: int = 45

    # -- Overlay --
    line_color: tuple = (0, 140, 255)  # BGR orange


# =====================================================================
#  CONSTANTS  (camera, physics, IMM — from hsv_imm_pipeline.py)
# =====================================================================

_GRAVITY      = 9.81
_BALL_RADIUS  = 0.036

_CAM_FOCAL_PX  = 1371.0
_CAM_CX        = 960.0
_CAM_CY        = 540.0
_CAM_POS_WORLD = np.array([-2.0, 0.0, 0.71])
_CAM_TILT_DEG  = 4.19

_BOUNCE_E_NORMAL     = 0.55
_BOUNCE_E_TANGENTIAL = 0.92

_UKF_ALPHA = 1e-3
_UKF_BETA  = 2.0
_UKF_KAPPA = 0.0

_MEAS_SIGMA_DET  = 3.0
_MEAS_SIGMA_HSV  = 8.0
_MEAS_SIGMA_NONE = 1e6

_IMM_TRANSITION = np.array([
    [0.95, 0.05],
    [0.15, 0.85],
])


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


# =====================================================================
#  HSV COLOUR RECOVERY
# =====================================================================

def _learn_hsv(frame, xyxy, cfg: TrackerConfig):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    x1 = _clip(x1, 0, w-1); x2 = _clip(x2, 0, w-1)
    y1 = _clip(y1, 0, h-1); y2 = _clip(y2, 0, h-1)
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
            ((0,    s_lo, v_lo), (h_hi, s_hi, v_hi)),
            ((h_lo, s_lo, v_lo), (179,  s_hi, v_hi)),
        ]
    return {"hue_ranges": ranges}


def _hsv_blob(frame, roi_xyxy, model, pred_xy, cfg: TrackerConfig):
    if model is None:
        return None
    H, W = frame.shape[:2]
    rx1, ry1, rx2, ry2 = map(int, roi_xyxy)
    rx1 = _clip(rx1, 0, W-1); rx2 = _clip(rx2, 0, W-1)
    ry1 = _clip(ry1, 0, H-1); ry2 = _clip(ry2, 0, H-1)
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern, iterations=cfg.hsv_morph_iters)
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
#  CAMERA MODEL
# =====================================================================

class CameraModel:
    """Pinhole camera for behind-the-stumps view."""

    def __init__(self):
        self.fx = _CAM_FOCAL_PX
        self.fy = _CAM_FOCAL_PX
        self.cx = _CAM_CX
        self.cy = _CAM_CY

        R_base = np.array([
            [0.0,  1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0],
        ])
        t_rad = np.radians(_CAM_TILT_DEG)
        R_tilt = np.array([
            [1.0, 0.0,            0.0           ],
            [0.0, np.cos(t_rad), -np.sin(t_rad) ],
            [0.0, np.sin(t_rad),  np.cos(t_rad) ],
        ])
        self.R = R_tilt @ R_base
        self.t = -self.R @ _CAM_POS_WORLD

    def project(self, p_world):
        """3D world point → 2D pixel (u, v)."""
        p_cam = self.R @ p_world + self.t
        if p_cam[2] <= 0.01:
            p_cam[2] = 0.01
        u = self.fx * p_cam[0] / p_cam[2] + self.cx
        v = self.fy * p_cam[1] / p_cam[2] + self.cy
        return np.array([u, v])

    def back_project_ray(self, u, v):
        """Pixel (u,v) → unit ray direction in world coords."""
        ray_cam = np.array([(u - self.cx)/self.fx, (v - self.cy)/self.fy, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = self.R.T @ ray_cam
        return ray_world / np.linalg.norm(ray_world)


# =====================================================================
#  UKF / IMM
# =====================================================================

def _sigma_points(x, P):
    """Scaled unscented transform sigma points."""
    n   = len(x)
    lam = _UKF_ALPHA**2 * (n + _UKF_KAPPA) - n
    try:
        S = cholesky((n + lam) * P, lower=True)
    except np.linalg.LinAlgError:
        S = cholesky((n + lam) * (P + 1e-8*np.eye(n)), lower=True)
    pts = np.zeros((2*n+1, n))
    pts[0] = x
    for i in range(n):
        pts[i+1]   = x + S[:, i]
        pts[n+i+1] = x - S[:, i]
    Wm = np.full(2*n+1, 1.0/(2*(n+lam)))
    Wc = np.full(2*n+1, 1.0/(2*(n+lam)))
    Wm[0] = lam/(n+lam)
    Wc[0] = lam/(n+lam) + (1 - _UKF_ALPHA**2 + _UKF_BETA)
    return pts, Wm, Wc


def _f_ballistic(x, dt):
    """Ballistic dynamics: constant velocity + gravity."""
    X, Y, Z, VX, VY, VZ = x
    return np.array([
        X + VX*dt,
        Y + VY*dt,
        Z + VZ*dt - 0.5*_GRAVITY*dt**2,
        VX,
        VY,
        VZ - _GRAVITY*dt,
    ])



class UKFMode:
    """One UKF instance for one IMM mode. Holds its own state, covariance, Q."""

    def __init__(self, name, f_func, Q, x0, P0, cam):
        self.name = name
        self.f    = f_func
        self.Q    = Q.copy()
        self.x    = x0.copy()
        self.P    = P0.copy()
        self.cam  = cam
        self.lik  = 1.0

    def predict(self, dt):
        sig, Wm, Wc = _sigma_points(self.x, self.P)
        sig_pred = np.array([self.f(sig[i], dt) for i in range(len(sig))])
        self.x = np.sum(Wm[:,None] * sig_pred, axis=0)
        self.P = self.Q.copy()
        for i in range(len(sig)):
            d = sig_pred[i] - self.x
            self.P += Wc[i] * np.outer(d, d)

    def update(self, z, R):
        """Kalman update with pixel measurement z=[u,v]."""
        sig, Wm, Wc = _sigma_points(self.x, self.P)
        z_sig  = np.array([self.cam.project(sig[i, :3]) for i in range(len(sig))])
        z_pred = np.sum(Wm[:,None] * z_sig, axis=0)
        S = R.copy(); Pxz = np.zeros((6, 2))
        for i in range(len(sig)):
            dz = z_sig[i] - z_pred; dx = sig[i] - self.x
            S   += Wc[i] * np.outer(dz, dz)
            Pxz += Wc[i] * np.outer(dx, dz)
        S_inv = np.linalg.inv(S)
        K     = Pxz @ S_inv
        innov = z - z_pred
        maha  = float(innov @ S_inv @ innov)
        self.x = self.x + K @ innov
        self.P = self.P - K @ S @ K.T
        self.P = 0.5*(self.P + self.P.T)
        det_S   = max(np.linalg.det(S), 1e-30)
        self.lik = float(np.exp(-0.5*maha) / np.sqrt((2*np.pi)**2 * det_S))
        return maha


class IMMFilter:
    """Two-mode IMM: ballistic + occluded (matches hsv_imm_pipeline.py)."""

    def __init__(self, cam, x0, P0, fps):
        self.cam = cam
        self.dt  = 1.0 / fps
        self.N   = 2
        self.mu  = np.array([0.92, 0.08])
        self.Ptr = _IMM_TRANSITION.copy()

        Q_flight = np.diag([0.01, 0.005, 0.005, 0.5,  0.3,  0.5 ])
        Q_coast  = np.diag([0.05, 0.02,  0.02,  2.0,  1.0,  2.0 ])

        self.modes = [
            UKFMode("ballistic", _f_ballistic, Q_flight, x0, P0, cam),
            UKFMode("occluded",  _f_ballistic, Q_coast,  x0, P0, cam),
        ]

    def _mix(self):
        c = np.zeros(self.N)
        for j in range(self.N):
            for i in range(self.N):
                c[j] += self.Ptr[i, j] * self.mu[i]
            c[j] = max(c[j], 1e-30)
        omega = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                omega[i, j] = self.Ptr[i, j] * self.mu[i] / c[j]
        for j in range(self.N):
            x_mix = sum(omega[i, j]*self.modes[i].x for i in range(self.N))
            P_mix = np.zeros((6, 6))
            for i in range(self.N):
                d = self.modes[i].x - x_mix
                P_mix += omega[i, j] * (self.modes[i].P + np.outer(d, d))
            self.modes[j].x = x_mix
            self.modes[j].P = P_mix
        return c

    def step(self, z, R, status, misses_count):
        c = self._mix()
        for m in self.modes:
            m.predict(self.dt)
        use_meas = (z is not None) and (status in ('det', 'hsv'))
        if use_meas:
            for m in self.modes:
                maha = m.update(z, R)
                if maha > 16.0: m.lik *= 0.01
        else:
            for m in self.modes: m.lik = 1.0
            if status in ('pred', 'miss'):
                self.mu[1] = min(0.95, self.mu[1] * 1.3)
            if misses_count > 5:
                self.mu[1] = min(0.95, self.mu[1] * 1.5)
        for j in range(self.N):
            self.mu[j] = self.modes[j].lik * c[j]
        total = self.mu.sum()
        if total > 1e-30: self.mu /= total
        else: self.mu = np.ones(self.N)/self.N

    def fused_state(self):
        return sum(self.mu[j]*self.modes[j].x for j in range(self.N))

    def speed_mps(self):
        x = self.fused_state()
        return float(np.sqrt(x[3]**2 + x[4]**2 + x[5]**2))

def _init_3d_state(obs_list, cam, fps):
    n   = min(6, len(obs_list))
    obs = obs_list[:n]
    u_avg = np.mean([o['u'] for o in obs])
    v_avg = np.mean([o['v'] for o in obs])
    ray   = cam.back_project_ray(u_avg, v_avg)
    t_ray = 18.0 / ray[0] if abs(ray[0]) > 1e-4 else 18.0
    p     = _CAM_POS_WORLD + t_ray * ray
    p[0]  = np.clip(p[0], 10.0, 22.0)
    p[1]  = np.clip(p[1], -2.0,  2.0)
    p[2]  = np.clip(p[2],  1.0,  2.8)
    speed = 35.0
    if n >= 2:
        du  = obs[-1]['u'] - obs[0]['u']
        dv  = obs[-1]['v'] - obs[0]['v']
        vz  = np.clip(-2.0 + (-dv/max(abs(du), 1))*3.0, -5.0, 5.0)
        vy  = np.clip((du/max(abs(du)+abs(dv), 1))*2.0, -3.0, 3.0)
        vx  = -np.sqrt(max(speed**2 - vy**2 - vz**2, 100.0))
    else:
        vx, vy, vz = -35.0, 0.0, -1.5
    x0 = np.array([p[0], p[1], p[2], vx, vy, vz])
    P0 = np.diag([4.0, 1.0, 0.5, 50.0, 10.0, 10.0])
    return x0, P0


# =====================================================================
#  TRAJECTORY FUNCTIONS
# =====================================================================

def _fit_parabola(x, y):
    if len(x) < 3:
        return None
    try:
        return np.polyfit(x, y, 2)
    except Exception:
        return None


def _clean_detections(x_arr, y_arr, frames_arr, status_arr):
    """
    Remove false-positive detections that corrupt the parabolic fits.

    Step 1: remove pre-delivery static false positives by detecting the first
    large spatial jump (>= 100 px) — everything before it is discarded.

    Step 2: remove unreliable HSV-only detections inside the bounce gap
    (the largest frame gap between consecutive YOLO detections) using a
    speed gate of 10–50 px/frame relative to the last confirmed YOLO hit.
    """
    n = len(x_arr)
    if n < 2:
        return x_arr, y_arr, frames_arr, status_arr

    # Step 1: remove pre-delivery false positives
    ONSET_JUMP_PX = 100.0
    jumps = np.sqrt(np.diff(x_arr)**2 + np.diff(y_arr)**2)
    onset_indices = np.where(jumps >= ONSET_JUMP_PX)[0]

    if len(onset_indices) > 0:
        start      = onset_indices[0] + 1
        x_arr      = x_arr[start:]
        y_arr      = y_arr[start:]
        frames_arr = frames_arr[start:]
        if status_arr is not None:
            status_arr = status_arr[start:]

    # Step 2: remove unreliable HSV detections in the bounce gap
    if status_arr is not None and len(x_arr) > 0:
        det_mask    = (status_arr == 'det') | (status_arr == 'detected')
        det_indices = np.where(det_mask)[0]

        if len(det_indices) >= 6:
            det_frames = frames_arr[det_indices]
            gaps       = np.diff(det_frames)

            if len(gaps) > 0:
                biggest        = int(np.argmax(gaps))
                gap_start_frame = det_frames[biggest]
                gap_end_frame   = det_frames[biggest + 1]

                if gaps[biggest] > 5:
                    ref_idx = det_indices[biggest]
                    x_ref   = x_arr[ref_idx]
                    y_ref   = y_arr[ref_idx]
                    f_ref   = frames_arr[ref_idx]

                    MIN_SPEED = 10.0  # px/frame
                    MAX_SPEED = 50.0  # px/frame

                    keep = np.ones(len(x_arr), dtype=bool)
                    for i in range(len(x_arr)):
                        in_gap = (frames_arr[i] > gap_start_frame and
                                  frames_arr[i] < gap_end_frame)
                        is_hsv = (status_arr[i] == 'hsv')
                        if in_gap and is_hsv:
                            frame_gap = frames_arr[i] - f_ref
                            if frame_gap > 0:
                                dist  = np.sqrt((x_arr[i] - x_ref)**2 +
                                                (y_arr[i] - y_ref)**2)
                                speed = dist / frame_gap
                                if not (MIN_SPEED <= speed <= MAX_SPEED):
                                    keep[i] = False
                            else:
                                keep[i] = False

                    x_arr      = x_arr[keep]
                    y_arr      = y_arr[keep]
                    frames_arr = frames_arr[keep]
                    status_arr = status_arr[keep]

    return x_arr, y_arr, frames_arr, status_arr


def _smooth_trajectory(x_raw, y_raw, min_points, bounce_thresh, status_arr=None):
    """Parabolic smoothing with bounce detection. Returns (x, y, bounce_pt)."""
    n = len(x_raw)
    if n < min_points:
        return x_raw.copy(), y_raw.copy(), None

    bounce_idx = None
    dy = np.diff(y_raw)
    for i in range(2, len(dy) - 1):
        if dy[i-1] > 0 and dy[i] < 0:
            local_range = y_raw[max(0, i-2):min(n, i+3)]
            if np.ptp(local_range) > bounce_thresh:
                bounce_idx = i
                break

    if bounce_idx is None:
        coeffs = _fit_parabola(x_raw, y_raw)
        if coeffs is not None:
            return x_raw.copy(), np.polyval(coeffs, x_raw), None
        return x_raw.copy(), y_raw.copy(), None

    x_pre, y_pre   = x_raw[:bounce_idx+1], y_raw[:bounce_idx+1]
    x_post, y_post = x_raw[bounce_idx:],   y_raw[bounce_idx:]

    coeffs_pre = _fit_parabola(x_pre, y_pre)

    # Post-bounce: weight HSV recoveries at 2× influence of YOLO dets
    # so noisy transition-zone points don't distort the fitted curve.
    if status_arr is not None and len(x_post) >= 3:
        status_post = status_arr[bounce_idx:]
        w_post = np.where(status_post == 'hsv', 2.0, 1.0)
        try:
            coeffs_post = np.polyfit(x_post, y_post, 2, w=w_post)
        except Exception:
            coeffs_post = _fit_parabola(x_post, y_post)
    else:
        coeffs_post = _fit_parabola(x_post, y_post)

    y_smooth_pre  = np.polyval(coeffs_pre,  x_pre)  if coeffs_pre  is not None else y_pre.copy()
    y_smooth_post = np.polyval(coeffs_post, x_post) if coeffs_post is not None else y_post.copy()

    x_smooth  = np.concatenate([x_pre, x_post[1:]])
    y_smooth  = np.concatenate([y_smooth_pre, y_smooth_post[1:]])
    bounce_pt = (x_raw[bounce_idx], float(y_smooth_pre[-1]))
    return x_smooth, y_smooth, bounce_pt


def _find_cutoff(x, y, bounce_idx, angle_cutoff):
    if bounce_idx is None:
        return len(x) - 1
    for i in range(bounce_idx + 2, len(x) - 1):
        v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
        v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 < 0.5 or m2 < 0.5:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1, 1)
        if np.degrees(np.arccos(cos_a)) > angle_cutoff:
            return i
    return len(x) - 1


def _load_trajectory(csv_path: str, cfg: TrackerConfig) -> dict:
    """Load and process trajectory CSV. Identical to trajectorywriter.load_trajectory."""
    df = pd.read_csv(csv_path)

    mask = df['x'].notna() & df['y'].notna()
    if 'status' in df.columns:
        mask &= df['status'].isin(['det', 'hsv', 'detected'])

    valid = df[mask].reset_index(drop=True)
    if len(valid) < cfg.traj_min_points:
        raise ValueError(f"Need at least {cfg.traj_min_points} points, got {len(valid)}")

    x_raw      = valid['x'].values.astype(float)
    y_raw      = valid['y'].values.astype(float)
    frames     = valid['frame'].values.astype(int)
    status_arr = valid['status'].values if 'status' in valid.columns else None

    # Remove false-positive detections before smoothing
    x_raw, y_raw, frames, status_arr = _clean_detections(x_raw, y_raw, frames, status_arr)
    if len(x_raw) < cfg.traj_min_points:
        raise ValueError(
            f"After cleaning, only {len(x_raw)} valid points remain (need {cfg.traj_min_points})")

    x_smooth, y_smooth, bounce_pt = _smooth_trajectory(
        x_raw, y_raw, cfg.traj_min_points, cfg.traj_bounce_thresh, status_arr)

    bounce_idx = None
    if bounce_pt is not None:
        dists      = (x_smooth - bounce_pt[0])**2 + (y_smooth - bounce_pt[1])**2
        bounce_idx = int(np.argmin(dists))

    cutoff  = _find_cutoff(x_smooth, y_smooth, bounce_idx, cfg.traj_angle_cutoff)
    x_final = x_smooth[:cutoff+1]
    y_final = y_smooth[:cutoff+1]

    pts = np.column_stack([x_final, y_final]).astype(np.int32)
    return {
        'points':      pts,
        'bounce':      bounce_pt,
        'start_frame': int(frames[0]),
        'end_frame':   int(frames[min(cutoff, len(frames)-1)]),
    }


# =====================================================================
#  IMM SPEED ESTIMATION
# =====================================================================

def _run_imm_speed(csv_rows: list, fps: float) -> dict:
    """Run IMM+UKF on in-memory CSV rows. Returns speed dict."""
    df = pd.DataFrame(csv_rows)
    if df.empty or 'x' not in df.columns:
        return {"imm_success": False}
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    mask = df['x'].notna() & df['y'].notna()
    if 'status' in df.columns:
        mask &= df['status'].isin(['det', 'hsv', 'detected'])
    valid = df[mask].reset_index(drop=True)
    if len(valid) < 5:
        return {"imm_success": False}

    cam = CameraModel()
    obs_list = [
        {'frame': int(r['frame']), 'u': float(r['x']), 'v': float(r['y']),
         'status': str(r.get('status', 'det')), 'misses': int(r.get('misses', 0))}
        for _, r in valid.iterrows()
    ]

    all_obs = {}
    for _, row in df.iterrows():
        fi  = int(row['frame'])
        has = (not pd.isna(row.get('x', np.nan))
               and str(row.get('status', '')) in ('det', 'hsv', 'detected'))
        all_obs[fi] = {
            'u':      float(row['x']) if has else None,
            'v':      float(row['y']) if has else None,
            'status': str(row.get('status', 'miss')),
            'misses': int(row.get('misses', 0)),
        }

    x0, P0  = _init_3d_state(obs_list, cam, fps)
    imm     = IMMFilter(cam, x0, P0, fps)
    first_f = obs_list[0]['frame']
    last_f  = obs_list[-1]['frame']
    speeds = []

    for fi in range(first_f, last_f + 1):
        o = all_obs.get(fi, {'u': None, 'v': None, 'status': 'miss', 'misses': 0})
        if o['u'] is not None:
            z   = np.array([o['u'], o['v']])
            sig = _MEAS_SIGMA_DET if o['status'] == 'det' else _MEAS_SIGMA_HSV
            R   = np.diag([sig**2, sig**2])
        else:
            z = None
            R = np.diag([_MEAS_SIGMA_NONE**2] * 2)
        imm.step(z, R, o['status'], o['misses'])
        speeds.append(imm.speed_mps())

    speeds = np.array(speeds) * 3.6   # m/s → km/h
    if len(speeds) >= 5:
        speeds = np.convolve(speeds, np.ones(5)/5, mode='same')

    win     = min(8, len(speeds))
    rel     = speeds[:win]
    med     = np.median(rel)
    clean   = rel[rel < 2.0 * med]
    release = float(np.median(clean)) if len(clean) > 0 else float(med)
    release = float(np.clip(release, 0.0, 165.0))

    return {
        "imm_success":       True,
        "release_speed_kmh": release,
    }


# =====================================================================
#  OVERLAY RENDERING
# =====================================================================

def _draw_overlay(frame, traj_data, color, release_kmh=None):
    """Draw trajectory line, bounce marker, and optional release speed text."""
    pts    = traj_data['points']
    bounce = traj_data['bounce']

    if len(pts) < 2:
        return frame

    cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

    if bounce is not None:
        cx, cy = int(bounce[0]), int(bounce[1])
        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy),  8, color,           2, cv2.LINE_AA)

    if release_kmh and release_kmh > 0:
        txt    = f"Release: {release_kmh:.0f} km/h"
        font   = cv2.FONT_HERSHEY_SIMPLEX
        sc, th = 0.9, 2
        pad    = 10
        tw, line_h = cv2.getTextSize(txt, font, sc, th)[0]
        line_h += 8
        ov = frame.copy()
        cv2.rectangle(ov, (pad, pad), (pad + tw + 2 * pad, pad + line_h + pad), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, txt, (2 * pad, pad + line_h),
                    font, sc, (255, 255, 255), th, cv2.LINE_AA)

    return frame


# =====================================================================
#  MAIN CLASS
# =====================================================================

class CricketBallTracker:
    """End-to-end cricket ball tracking and trajectory rendering."""

    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self._model: YOLO | None = None

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
            bounce, fps, release_speed_kmh, bounce_speed_kmh (always None).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _cb(cur, tot, msg):
            if progress_callback:
                progress_callback(cur, tot, msg)

        # Pass 1 — YOLO + ByteTrack + HSV detection
        csv_rows, fps, boxes_by_frame = self._detection_pass(video_path, out, _cb)

        # Pass 2 — IMM speed + parabolic trajectory + overlay rendering
        traj_video  = None
        bounce      = None
        release_kmh = None

        try:
            traj_video, bounce, release_kmh = self._rendering_pass(
                video_path, out, csv_rows, fps, boxes_by_frame, _cb)
        except Exception as e:
            _cb(0, 0, f"Pass 2 skipped: {e}")

        return {
            "detection_video":   None,
            "trajectory_video":  str(traj_video) if traj_video else None,
            "csv":               str(out / "trajectory.csv"),
            "bounce":            bounce,
            "fps":               fps,
            "release_speed_kmh": release_kmh,
            "bounce_speed_kmh":  None,  # 2-mode IMM does not compute bounce speed
        }

    # ------------------------------------------------------------------
    #  Pass 1 — Detection
    # ------------------------------------------------------------------

    def _detection_pass(self, video_path, out_dir, cb):
        cfg = self.cfg
        tracker_yaml = out_dir / "bytetrack_custom.yaml"
        _write_bytetrack_yaml(tracker_yaml, cfg)

        model  = self._get_model()
        stream = model.track(
            source=video_path, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
            device=cfg.device, classes=cfg.classes,
            tracker=str(tracker_yaml), persist=True, stream=True,
            verbose=False, save=False,
        )

        csv_rows      = []
        boxes_by_frame = {}
        fps           = 30.0
        active_id     = None
        last_pos      = None
        vel           = np.array([0.0, 0.0], dtype=float)
        misses        = 0
        hsv_model     = None

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

            candidates = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy_arr = result.boxes.xyxy.cpu().numpy()
                confs    = result.boxes.conf.cpu().numpy()
                ids      = None
                if getattr(result.boxes, "id", None) is not None:
                    ids = result.boxes.id.cpu().numpy().astype(int)
                for i in range(len(xyxy_arr)):
                    x1, y1, x2, y2 = xyxy_arr[i]
                    if cfg.min_box_area_px > 0 and _area(x1, y1, x2, y2) < cfg.min_box_area_px:
                        continue
                    candidates.append({
                        "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                        "conf": float(confs[i]),
                        "id":   int(ids[i]) if ids is not None else None,
                    })

            pred = None
            if last_pos is not None:
                pred = (last_pos[0] + float(vel[0]), last_pos[1] + float(vel[1]))

            chosen = None
            gate   = cfg.gate_radius_px + misses * cfg.gate_grow_per_miss

            if active_id is None:
                if candidates:
                    chosen    = max(candidates, key=lambda d: d["conf"])
                    active_id = chosen["id"]
            else:
                same_id = [d for d in candidates if d["id"] == active_id]
                pool    = same_id if same_id else candidates
                if pool and pred is not None:
                    best = min(pool, key=lambda d: math.hypot(
                        _center(*d["xyxy"])[0] - pred[0],
                        _center(*d["xyxy"])[1] - pred[1]))
                    if math.hypot(_center(*best["xyxy"])[0] - pred[0],
                                  _center(*best["xyxy"])[1] - pred[1]) <= gate:
                        chosen = best
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
                r    = _roi(pred[0], pred[1], half, fw, fh)
                bb   = _hsv_blob(frame, r, hsv_model, pred, cfg)
                if bb is not None:
                    chosen = {"xyxy": bb, "conf": 0.0, "id": active_id}
                    hsv_recovered = True

            # State update
            status = "miss"
            if chosen is not None:
                cx, cy = _center(*chosen["xyxy"])
                if last_pos is not None:
                    new_vel = np.array([cx - last_pos[0], cy - last_pos[1]], dtype=float)
                    vel     = (1 - cfg.vel_ema_alpha) * vel + cfg.vel_ema_alpha * new_vel
                last_pos  = (cx, cy)
                misses    = 0
                status    = "hsv" if hsv_recovered else "det"
                if chosen["id"] is not None:
                    active_id = chosen["id"]
            else:
                misses += 1
                status  = "pred" if pred is not None else "miss"
                if misses > cfg.max_misses:
                    active_id = None; last_pos = None; vel[:] = 0.0
                    hsv_model = None; misses = 0
                    status    = "reset"

            # Record bounding box for trajectory-video overlay
            if chosen is not None:
                boxes_by_frame[fi] = (chosen["xyxy"], hsv_recovered)

            csv_rows.append({
                "frame":  fi,
                "status": status,
                "id":     active_id if active_id is not None else "",
                "x":      last_pos[0] if last_pos is not None else "",
                "y":      last_pos[1] if last_pos is not None else "",
                "vx":     float(vel[0]),
                "vy":     float(vel[1]),
                "gate":   gate,
                "misses": misses,
            })

            if fi % 10 == 0:
                cb(fi, total_frames, "Pass 1: detecting ball")

        # Persist CSV for bucket upload and debugging — always write, even with zero rows
        _CSV_FIELDS = ["frame", "status", "id", "x", "y", "vx", "vy", "gate", "misses"]
        csv_path = out_dir / "trajectory.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fields = list(csv_rows[0].keys()) if csv_rows else _CSV_FIELDS
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(csv_rows)

        cb(total_frames, total_frames, "Pass 1: complete")
        return csv_rows, fps, boxes_by_frame

    # ------------------------------------------------------------------
    #  Pass 2 — Speed estimation + trajectory rendering
    # ------------------------------------------------------------------

    def _rendering_pass(self, video_path, out_dir, csv_rows, fps, boxes_by_frame, cb):
        cfg = self.cfg

        # IMM + UKF speed estimation
        imm_res     = _run_imm_speed(csv_rows, fps)
        release_kmh = imm_res.get("release_speed_kmh") if imm_res.get("imm_success") else None

        # Parabolic trajectory smoothing — reads from the written CSV (identical to trajectorywriter.py)
        traj = _load_trajectory(str(out_dir / "trajectory.csv"), cfg)

        # Render trajectory overlay on the original input video (avoids mp4v read-back issues on Linux)
        output_vid = out_dir / "trajectory_output.mp4"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open input video: {video_path}")

        fps_v  = cap.get(cv2.CAP_PROP_FPS) or fps
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        writer = cv2.VideoWriter(
            str(output_vid), cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (width, height))

        cb(0, total, "Pass 2: rendering trajectory")
        fn = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fn in boxes_by_frame:
                xyxy, hsv_rec = boxes_by_frame[fn]
                bx1, by1, bx2, by2 = map(int, xyxy)
                col = (0, 255, 255) if hsv_rec else (0, 255, 0)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)
            if fn >= traj['start_frame']:
                _draw_overlay(frame, traj, cfg.line_color, release_kmh)
            writer.write(frame)
            fn += 1
            if fn % 10 == 0:
                cb(fn, total, "Pass 2: rendering trajectory")

        cap.release()
        writer.release()
        cb(total, total, "Pass 2: complete")

        return output_vid, traj.get('bounce'), release_kmh
