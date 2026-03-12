#!/usr/bin/env python3
"""
Cricket ball trajectory overlay.
Reads tracking CSV, applies physics-based smoothing, marks pitch point.
"""

import cv2
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION - edit these
# =============================================================================
INPUT_VIDEO = 'run_output/annotated_output.mp4'
TRAJECTORY_CSV = 'run_output/trajectory.csv'
OUTPUT_VIDEO = 'outputtraj.mp4'
LINE_COLOR = (0, 140, 255)  # BGR - orange works well on most pitches

# tunables
MIN_POINTS = 5
BOUNCE_THRESHOLD = 8   # min y-change to count as real bounce
ANGLE_CUTOFF = 45      # degrees - cuts trajectory at bat contact
# =============================================================================


def fit_parabola(x, y):
    """
    Fit parabola through points. Physics: ball under gravity follows parabola.
    Returns coefficients for y = ax^2 + bx + c
    """
    if len(x) < 3:
        return None
    try:
        return np.polyfit(x, y, 2)
    except:
        return None


def smooth_trajectory_physics(x_raw, y_raw):
    """
    Smooth using parabolic fits - matches real ball physics.
    Cricket ball: parabola down, bounce, parabola up then down.
    """
    n = len(x_raw)
    if n < MIN_POINTS:
        return x_raw.copy(), y_raw.copy(), None
    
    # find bounce - where vertical velocity flips
    # (image coords: y increases downward, so falling = positive dy)
    bounce_idx = None
    dy = np.diff(y_raw)
    
    for i in range(2, len(dy) - 1):
        # was falling, now rising
        if dy[i-1] > 0 and dy[i] < 0:
            # check it's not just noise
            local_range = y_raw[max(0, i-2):min(n, i+3)]
            if np.ptp(local_range) > BOUNCE_THRESHOLD:
                bounce_idx = i
                break
    
    # no bounce - fit single parabola
    if bounce_idx is None:
        coeffs = fit_parabola(x_raw, y_raw)
        if coeffs is not None:
            y_smooth = np.polyval(coeffs, x_raw)
            return x_raw.copy(), y_smooth, None
        return x_raw.copy(), y_raw.copy(), None
    
    # fit pre-bounce and post-bounce separately
    x_pre, y_pre = x_raw[:bounce_idx+1], y_raw[:bounce_idx+1]
    x_post, y_post = x_raw[bounce_idx:], y_raw[bounce_idx:]
    
    coeffs_pre = fit_parabola(x_pre, y_pre)
    coeffs_post = fit_parabola(x_post, y_post)
    
    if coeffs_pre is not None:
        y_smooth_pre = np.polyval(coeffs_pre, x_pre)
    else:
        y_smooth_pre = y_pre.copy()
    
    if coeffs_post is not None:
        y_smooth_post = np.polyval(coeffs_post, x_post)
    else:
        y_smooth_post = y_post.copy()
    
    # stitch together
    x_smooth = np.concatenate([x_pre, x_post[1:]])
    y_smooth = np.concatenate([y_smooth_pre, y_smooth_post[1:]])
    
    bounce_pt = (x_raw[bounce_idx], y_smooth_pre[-1])
    
    return x_smooth, y_smooth, bounce_pt


def find_cutoff(x, y, bounce_idx):
    """
    Find where to cut trajectory - usually bat contact.
    Looks for sharp direction change after bounce.
    """
    if bounce_idx is None:
        return len(x) - 1
    
    for i in range(bounce_idx + 2, len(x) - 1):
        v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
        v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
        
        mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if mag1 < 0.5 or mag2 < 0.5:
            continue
        
        cos_ang = np.clip(np.dot(v1, v2) / (mag1 * mag2), -1, 1)
        angle = np.degrees(np.arccos(cos_ang))
        
        if angle > ANGLE_CUTOFF:
            return i
    
    return len(x) - 1


def load_trajectory(csv_path):
    """
    Load and process trajectory CSV.
    Expects columns: frame, x, y, status
    """
    df = pd.read_csv(csv_path)
    
    # only actual detections
    mask = df['x'].notna() & df['y'].notna()
    if 'status' in df.columns:
        mask &= df['status'].isin(['det', 'hsv', 'detected'])
    
    valid = df[mask].reset_index(drop=True)
    
    if len(valid) < MIN_POINTS:
        raise ValueError(f"Need at least {MIN_POINTS} points, got {len(valid)}")
    
    x_raw = valid['x'].values.astype(float)
    y_raw = valid['y'].values.astype(float)
    frames = valid['frame'].values.astype(int)
    
    # physics smoothing
    x_smooth, y_smooth, bounce_pt = smooth_trajectory_physics(x_raw, y_raw)
    
    # find bounce index in smoothed data
    bounce_idx = None
    if bounce_pt is not None:
        dists = (x_smooth - bounce_pt[0])**2 + (y_smooth - bounce_pt[1])**2
        bounce_idx = np.argmin(dists)
    
    # cut at bat contact
    cutoff = find_cutoff(x_smooth, y_smooth, bounce_idx)
    
    x_final = x_smooth[:cutoff+1]
    y_final = y_smooth[:cutoff+1]
    
    pts = np.column_stack([x_final, y_final]).astype(np.int32)
    
    return {
        'points': pts,
        'bounce': bounce_pt,
        'start_frame': int(frames[0]),
        'end_frame': int(frames[min(cutoff, len(frames)-1)])
    }


def draw_overlay(frame, traj_data, color):
    """
    Draw trajectory and bounce marker. Clean, broadcast style.
    """
    pts = traj_data['points']
    bounce = traj_data['bounce']
    
    if len(pts) < 2:
        return frame
    
    # trajectory line - simple and clean
    cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)
    
    # bounce marker - two rings
    if bounce is not None:
        cx, cy = int(bounce[0]), int(bounce[1])
        cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 8, color, 2, cv2.LINE_AA)
    
    return frame


def process_video():
    """
    Main loop - overlay trajectory on video.
    """
    print(f"Loading: {TRAJECTORY_CSV}")
    traj = load_trajectory(TRAJECTORY_CSV)
    print(f"  {len(traj['points'])} points, frames {traj['start_frame']}-{traj['end_frame']}")
    if traj['bounce']:
        print(f"  Bounce at ({traj['bounce'][0]:.1f}, {traj['bounce'][1]:.1f})")
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise IOError(f"Can't open: {INPUT_VIDEO}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {w}x{h}, {fps:.1f}fps, {total} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num >= traj['start_frame']:
            frame = draw_overlay(frame, traj, LINE_COLOR)
        
        out.write(frame)
        frame_num += 1
        
        if frame_num % 50 == 0:
            print(f"  {100*frame_num/total:.0f}%", end='\r')
    
    cap.release()
    out.release()
    print(f"\nSaved: {OUTPUT_VIDEO}")


if __name__ == '__main__':
    process_video()