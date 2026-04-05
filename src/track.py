"""
src/track.py - Phase 5b: Tracking (DeepSORT for balls AND robots)

Functions:
  run_ball_tracker()          - DeepSORT on ball detections
  run_robot_tracker()         - DeepSORT on robot detections
  build_robot_identity_map()  - maps track_id -> team_number via OCR
  calibrate_robot_identities()- sample first 300 frames, majority-vote OCR,
                                  confirm with user, save match_identity.json

Depends on: src/detect.py output (list of per-frame detection dicts).

(!) STOP after calibrate_robot_identities() prints the identity map --
   user must confirm team numbers before possession / attribution runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment


# ── ByteTrack-style two-stage IoU tracker ─────────────────────────────────────
# ByteTrack improves over SORT by cascading matching in two stages:
#   Stage 1 — match high-confidence detections against active tracks (IoU)
#   Stage 2 — match remaining active tracks against low-confidence detections
# This recovers tracks that SORT drops during occlusion/partial visibility,
# which is common when FRC robots pass in front of each other.

_HIGH_CONF = 0.55   # threshold separating high vs low confidence detections
_LOW_CONF  = 0.10   # minimum confidence to keep as a low-conf detection


def _iou(a: list, b: list) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def _iou_cost(tracks: list, boxes: list[list]) -> np.ndarray:
    """Build an (n_tracks × n_dets) IoU cost matrix (1 - IoU)."""
    n, m = len(tracks), len(boxes)
    cost = np.zeros((n, m), dtype=np.float64)
    for i, trk in enumerate(tracks):
        for j, det in enumerate(boxes):
            cost[i, j] = 1.0 - _iou(trk.bbox, det)
    return cost


def _match(tracks: list, boxes: list[list], cls_list: list[str],
           iou_threshold: float) -> tuple[set, set, set, list[tuple[int, int]]]:
    """
    Hungarian matching on IoU cost matrix.

    Returns:
        (matched_trk_indices, matched_det_indices, unmatched_det_indices,
         match_pairs [(trk_idx, det_idx), ...])
    """
    matched_trk: set[int] = set()
    matched_det: set[int] = set()
    pairs: list[tuple[int, int]] = []

    if not tracks or not boxes:
        return matched_trk, matched_det, set(range(len(boxes))), pairs

    cost   = _iou_cost(tracks, boxes)
    r_idx, c_idx = linear_sum_assignment(cost)
    for r, c in zip(r_idx, c_idx):
        if cost[r, c] <= (1.0 - iou_threshold):
            tracks[r].bbox             = boxes[c]
            tracks[r].cls              = cls_list[c]
            tracks[r].hits            += 1
            tracks[r].time_since_update = 0
            matched_trk.add(r)
            matched_det.add(c)
            pairs.append((r, c))

    unmatched_det = {j for j in range(len(boxes)) if j not in matched_det}
    return matched_trk, matched_det, unmatched_det, pairs


def _make_kalman(bbox: list):
    """
    Create a Kalman filter for a bounding box (#4).

    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement:  [cx, cy, w, h]
    """
    try:
        from filterpy.kalman import KalmanFilter
        kf = KalmanFilter(dim_x=8, dim_z=4)
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        kf.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=np.float64)
        # State transition: constant velocity
        kf.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            kf.F[i, i + 4] = 1.0
        # Measurement matrix: observe position/size only
        kf.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            kf.H[i, i] = 1.0
        kf.R  *= 10.0    # measurement noise
        kf.P  *= 10.0    # initial covariance
        kf.Q  *= 0.01    # process noise
        return kf
    except ImportError:
        return None


class _ByteTrack:
    __slots__ = ("track_id", "bbox", "cls", "hits", "time_since_update", "state", "_kf")
    # state: "active" | "lost"

    def __init__(self, track_id: int, bbox: list, cls: str):
        self.track_id          = track_id
        self.bbox              = bbox
        self.cls               = cls
        self.hits              = 1
        self.time_since_update = 0
        self.state             = "active"
        self._kf               = _make_kalman(bbox)  # Kalman filter (#4)

    def predict(self) -> list:
        """Predict next bbox position using Kalman filter (#4)."""
        if self._kf is not None:
            self._kf.predict()
            cx, cy, w, h = self._kf.x[:4]
            return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        return list(self.bbox)

    def update(self, bbox: list):
        """Update Kalman filter with new measurement (#4)."""
        self.bbox = bbox
        if self._kf is not None:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w  = bbox[2] - bbox[0]
            h  = bbox[3] - bbox[1]
            self._kf.update(np.array([cx, cy, w, h]))
            # Use Kalman-smoothed bbox as the new bbox
            cx2, cy2, w2, h2 = self._kf.x[:4]
            self.bbox = [cx2 - w2/2, cy2 - h2/2, cx2 + w2/2, cy2 + h2/2]

    def is_confirmed(self) -> bool:
        return self.hits >= 3

    def to_ltrb(self) -> list:
        return list(self.bbox)

    @property
    def det_class(self) -> str:
        return self.cls


class _ByteTracker:
    """
    ByteTrack-style two-stage cascade IoU tracker.

    Stage 1: high-conf detections matched against active tracks.
    Stage 2: lost/unmatched tracks matched against low-conf detections.
    New tracks spawned only from unmatched high-conf detections.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.25):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self._tracks: list[_ByteTrack] = []
        self._next_id = 1

    def update_tracks(self, detections: list) -> list[_ByteTrack]:
        """
        Args:
            detections: [([x1,y1,w,h], confidence, class_name), ...]
        Returns:
            List of active _ByteTrack objects (confirmed + unconfirmed).
        """
        # Split into high/low confidence
        hi_boxes: list[list] = []
        hi_cls:   list[str]  = []
        lo_boxes: list[list] = []
        lo_cls:   list[str]  = []

        for (x1, y1, w, h), conf, cls in detections:
            box = [x1, y1, x1 + w, y1 + h]
            if conf >= _HIGH_CONF:
                hi_boxes.append(box); hi_cls.append(cls)
            elif conf >= _LOW_CONF:
                lo_boxes.append(box); lo_cls.append(cls)

        active = [t for t in self._tracks if t.state == "active"]
        lost   = [t for t in self._tracks if t.state == "lost"]

        # Kalman predict step: advance each track's position estimate (#4)
        for trk in self._tracks:
            trk.bbox = trk.predict()

        # ── Stage 1: match high-conf dets against active tracks ───────────────
        matched_a, matched_h, unmatched_h, pairs_a = _match(
            active, hi_boxes, hi_cls, self.iou_threshold)

        # Build trk_idx → det_idx map from the returned pairs
        a_match_map: dict[int, int] = {r: c for r, c in pairs_a}

        # Update matched tracks with Kalman filter (#4)
        for i, trk in enumerate(active):
            if i in a_match_map:
                trk.update(hi_boxes[a_match_map[i]])  # Kalman update with correct detection
            else:
                trk.time_since_update += 1
                trk.state = "lost"

        # ── Stage 2: match unmatched high-conf dets AND lost tracks ───────────
        unmatched_h_boxes = [hi_boxes[j] for j in sorted(unmatched_h)]
        unmatched_h_cls   = [hi_cls[j]   for j in sorted(unmatched_h)]

        second_pool = lost

        if second_pool and lo_boxes:
            matched_l, matched_lo, _, _pairs_l = _match(
                second_pool, lo_boxes, lo_cls, self.iou_threshold)
            for i in matched_l:
                second_pool[i].state = "active"
                second_pool[i].time_since_update = 0
        # (unmatched low-conf dets discarded — not reliable enough for new tracks)

        # ── Spawn new tracks from unmatched high-conf dets only ───────────────
        for box, cls in zip(unmatched_h_boxes, unmatched_h_cls):
            self._tracks.append(_ByteTrack(self._next_id, box, cls))
            self._next_id += 1

        # ── Prune expired lost tracks ─────────────────────────────────────────
        self._tracks = [
            t for t in self._tracks
            if t.state == "active" or t.time_since_update <= self.max_age
        ]

        return list(self._tracks)


def _make_tracker(max_age: int = 30, n_init: int = 3) -> _ByteTracker:
    """Return a ByteTracker instance."""
    return _ByteTracker(max_age=max_age, min_hits=n_init)


# ---- NMS helper --------------------------------------------------------------

def _nms_dets(detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    """
    Non-maximum suppression on raw detections for a single frame.
    Removes overlapping boxes, keeping the highest-confidence one.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: list[dict] = []
    for d in dets:
        suppressed = False
        for k in kept:
            if _iou(d["bbox"], k["bbox"]) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(d)
    return kept


# ---- Detection format helpers -----------------------------------------------

def _dets_to_deepsort(detections: list[dict]) -> list[tuple]:
    """
    Convert our detection dicts to DeepSORT input format.

    DeepSORT expects: [([left, top, width, height], confidence, class_name), ...]
    """
    ds_input = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        w = x2 - x1
        h = y2 - y1
        ds_input.append(([x1, y1, w, h], d["confidence"], d["class_name"]))
    return ds_input


def _tracks_to_dicts(tracks, frame_id: int, source: str = "ball") -> list[dict]:
    """
    Convert DeepSORT Track objects to our standard dict format.

    Returns:
        [{frame_id, track_id, bbox [x1,y1,x2,y2], class_name, is_confirmed}]
    """
    result = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        ltrb = track.to_ltrb()
        result.append({
            "frame_id":    frame_id,
            "track_id":    track.track_id,
            "bbox":        [float(v) for v in ltrb],
            "class_name":  getattr(track, "det_class", source),
            "is_confirmed": True,
        })
    return result


# ---- Ball tracker -----------------------------------------------------------

def run_ball_tracker(
    all_frame_detections: list[dict],
    max_age:         int   = 4,    # drop lost tracks after 4 missed frames (fast decay)
    n_init:          int   = 1,    # confirm immediately; confidence threshold is the gate
    min_track_len:   int   = 3,    # discard tracks seen fewer than 3 times
    max_match_dist:  float = 250.0,  # px — centre-to-centre distance for matching
) -> dict[int, list[dict]]:
    """
    Lightweight ball tracker using centre-distance matching rather than IoU.

    IoU fails for fast balls sampled every N frames (ball moves far between
    samples, so consecutive boxes don't overlap at all). Distance matching
    tolerates large inter-frame jumps while still connecting the same ball.

    Post-filters applied after tracking:
      • min_track_len  — drop very short tracks (noise / single flickers)
      • min_displacement — track centre must move ≥ 30 px total (static = ground ball)

    Args:
        all_frame_detections: per-frame detection dicts from detect.process_video().
        max_age:        Sampled frames a track can be missing before deletion.
        n_init:         Detections needed to confirm a track (1 = immediate).
        min_track_len:  Minimum confirmed frames to keep a track.
        max_match_dist: Maximum centre distance (px) to associate a detection
                        with an existing track.

    Returns:
        {track_id: [{frame_id, track_id, bbox, class_name}]}
    """
    import math

    ball_classes = {"Fuel", "fuel", "ball", "Ball", "game_piece", "coral", "Coral"}

    # ── simple distance-based tracker state ──────────────────────────────────
    # Each track: {id, cx, cy, age, hits, entries}
    next_id  = 1
    live: list[dict] = []           # currently active tracks
    all_tracks: dict[int, list[dict]] = {}

    frames_sorted = sorted(all_frame_detections, key=lambda f: f["frame_id"])

    for frame_data in frames_sorted:
        fid  = frame_data["frame_id"]
        dets = [d for d in frame_data.get("detections", [])
                if d["class_name"] in ball_classes
                or d["class_name"].lower() in {"fuel", "ball", "game_piece",
                                               "coral"}]

        # Age all live tracks (+1 miss)
        for trk in live:
            trk["age"] += 1

        # Greedy nearest-neighbour match (sufficient for ball tracking)
        unmatched_dets = list(range(len(dets)))
        if live and dets:
            # Build distance matrix
            dist_mat = np.full((len(live), len(dets)), np.inf)
            for ti, trk in enumerate(live):
                for di, det in enumerate(dets):
                    cx = (det["bbox"][0] + det["bbox"][2]) / 2
                    cy = (det["bbox"][1] + det["bbox"][3]) / 2
                    dist_mat[ti, di] = math.sqrt(
                        (cx - trk["cx"]) ** 2 + (cy - trk["cy"]) ** 2
                    )

            matched_det: set[int] = set()
            # Greedy: assign closest unmatched det to each track (row order = track order)
            row_order = np.argsort(dist_mat.min(axis=1))  # tracks with closest det first
            for ti in row_order:
                best_di = int(np.argmin(dist_mat[ti]))
                if dist_mat[ti, best_di] <= max_match_dist and best_di not in matched_det:
                    det  = dets[best_di]
                    trk  = live[ti]
                    cx   = (det["bbox"][0] + det["bbox"][2]) / 2
                    cy   = (det["bbox"][1] + det["bbox"][3]) / 2
                    trk["cx"] = cx; trk["cy"] = cy
                    trk["age"] = 0
                    trk["hits"] += 1
                    entry = {"frame_id": fid, "track_id": trk["id"],
                             "bbox": det["bbox"], "class_name": det["class_name"]}
                    trk["entries"].append(entry)
                    all_tracks.setdefault(trk["id"], []).append(entry)
                    matched_det.add(best_di)
            unmatched_dets = [i for i in range(len(dets)) if i not in matched_det]

        # Spawn new tracks for unmatched detections
        for di in unmatched_dets:
            det = dets[di]
            cx  = (det["bbox"][0] + det["bbox"][2]) / 2
            cy  = (det["bbox"][1] + det["bbox"][3]) / 2
            tid = next_id; next_id += 1
            entry = {"frame_id": fid, "track_id": tid,
                     "bbox": det["bbox"], "class_name": det["class_name"]}
            live.append({"id": tid, "cx": cx, "cy": cy, "age": 0,
                         "hits": 1, "entries": [entry]})
            all_tracks[tid] = [entry]

        # Drop expired tracks
        live = [t for t in live if t["age"] <= max_age]

    # ── Post-filters ─────────────────────────────────────────────────────────
    # 1. Minimum track length
    all_tracks = {tid: ents for tid, ents in all_tracks.items()
                  if len(ents) >= min_track_len}

    # 2. Must have moved ≥ 30 px in total (eliminates static/ground balls)
    def _total_disp(ents):
        if len(ents) < 2:
            return 0.0
        x0 = (ents[0]["bbox"][0] + ents[0]["bbox"][2]) / 2
        y0 = (ents[0]["bbox"][1] + ents[0]["bbox"][3]) / 2
        x1 = (ents[-1]["bbox"][0] + ents[-1]["bbox"][2]) / 2
        y1 = (ents[-1]["bbox"][1] + ents[-1]["bbox"][3]) / 2
        return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    all_tracks = {tid: ents for tid, ents in all_tracks.items()
                  if _total_disp(ents) >= 30.0}

    print(f"  [Track] Ball tracker: {len(all_tracks)} valid ball tracks "
          f"(min {min_track_len} frames + >=30px displacement)")
    return all_tracks


# ---- Robot tracker ----------------------------------------------------------

_MAX_ROBOTS = 6   # FRC: 3 red + 3 blue, always exactly 6 robots on field


def run_robot_tracker(
    all_frame_detections: list[dict],
    max_age:    int = 8,    # sampled frames before dropping lost track (was 90)
    n_init:     int = 3,    # hits to confirm (was 5)
    max_robots: int = _MAX_ROBOTS,
) -> dict[int, list[dict]]:
    """
    Track robots using distance-based matching (same approach as ball tracker).

    IoU matching fails at high sample intervals because robots can move far
    between sampled frames. Distance matching tolerates larger inter-frame jumps.

    Strategy:
      1. NMS per frame (remove overlapping YOLO boxes).
      2. Keep top max_robots+2 detections by confidence.
      3. Distance-based greedy matching (max 300px centre-to-centre).
      4. Drop tracks that disappear quickly (< min_frames).
      5. Deduplicate overlapping fragments.
      6. Hard limit: top max_robots by longevity.
    """
    import math

    robot_classes = {"robot", "Robot", "Blue_Robot", "Red_Robot",
                     "blue_robot", "red_robot"}

    MAX_MATCH_DIST_ROBOT = 300.0   # robots are large, can shift 150-250px between samples

    next_id  = 1
    live: list[dict] = []
    all_tracks: dict[int, list[dict]] = {}

    frames_sorted = sorted(all_frame_detections, key=lambda f: f["frame_id"])

    for frame_data in frames_sorted:
        fid  = frame_data["frame_id"]
        dets = [d for d in frame_data.get("detections", [])
                if d["class_name"] in robot_classes]

        # NMS + cap
        dets = _nms_dets(dets, iou_threshold=0.45)
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)[: max_robots + 2]

        # Age all live tracks
        for trk in live:
            trk["age"] += 1

        # Distance-based matching
        unmatched_dets = list(range(len(dets)))
        if live and dets:
            dist_mat = np.full((len(live), len(dets)), np.inf)
            for ti, trk in enumerate(live):
                for di, det in enumerate(dets):
                    cx = (det["bbox"][0] + det["bbox"][2]) / 2
                    cy = (det["bbox"][1] + det["bbox"][3]) / 2
                    dist_mat[ti, di] = math.sqrt(
                        (cx - trk["cx"]) ** 2 + (cy - trk["cy"]) ** 2
                    )

            matched_det: set[int] = set()
            for ti in np.argsort(dist_mat.min(axis=1)):
                best_di = int(np.argmin(dist_mat[ti]))
                if dist_mat[ti, best_di] <= MAX_MATCH_DIST_ROBOT and best_di not in matched_det:
                    det = dets[best_di]
                    trk = live[ti]
                    trk["cx"]   = (det["bbox"][0] + det["bbox"][2]) / 2
                    trk["cy"]   = (det["bbox"][1] + det["bbox"][3]) / 2
                    trk["age"]  = 0
                    trk["hits"] += 1
                    entry = {"frame_id": fid, "track_id": trk["id"],
                             "bbox": det["bbox"], "class_name": "robot"}
                    trk["entries"].append(entry)
                    all_tracks.setdefault(trk["id"], []).append(entry)
                    matched_det.add(best_di)
            unmatched_dets = [i for i in range(len(dets)) if i not in matched_det]

        # Spawn new tracks for unmatched detections
        for di in unmatched_dets:
            det = dets[di]
            tid = next_id; next_id += 1
            cx  = (det["bbox"][0] + det["bbox"][2]) / 2
            cy  = (det["bbox"][1] + det["bbox"][3]) / 2
            entry = {"frame_id": fid, "track_id": tid,
                     "bbox": det["bbox"], "class_name": "robot"}
            live.append({"id": tid, "cx": cx, "cy": cy, "age": 0,
                         "hits": 1, "entries": [entry]})
            all_tracks[tid] = [entry]

        # Drop expired tracks
        live = [t for t in live if t["age"] <= max_age]

    # Drop short-lived noise tracks
    min_frames = max(8, len(all_frame_detections) // 60)
    all_tracks = {tid: ents for tid, ents in all_tracks.items()
                  if len(ents) >= min_frames}

    # Merge overlapping fragments from re-ID gaps
    print(f"  [Track] Robot tracker: {len(all_tracks)} tracks (before dedup)")
    all_tracks = deduplicate_robot_tracks(all_tracks)
    print(f"  [Track] Robot tracker: {len(all_tracks)} tracks (after dedup)")

    # Hard limit to exactly max_robots
    if len(all_tracks) > max_robots:
        top = sorted(all_tracks.items(), key=lambda kv: len(kv[1]), reverse=True)
        all_tracks = dict(top[:max_robots])
        print(f"  [Track] Keeping top {max_robots} by longevity -> {list(all_tracks.keys())}")

    print(f"  [Track] Final robot count: {len(all_tracks)} (expected {max_robots})")
    return all_tracks


# ---- Robot track deduplication -----------------------------------------------

def deduplicate_robot_tracks(
    robot_tracks: dict[int, list[dict]],
    max_gap_frames: int = 300,
    max_centre_dist_px: float = 200.0,
) -> dict[int, list[dict]]:
    """
    Merge robot tracks that are almost certainly the same robot re-acquired
    after a brief disappearance (occlusion, boundary exit, missed detection).

    Merging criteria (both must hold):
      1. The two tracks do NOT overlap in time (no shared frame_id).
      2. The gap between end of track A and start of track B is <= max_gap_frames.
      3. The last known position of A is within max_centre_dist_px of the first
         known position of B.

    The surviving track ID is the lower of the two (earlier track wins).
    Merged entries are sorted by frame_id.

    Args:
        robot_tracks:       {track_id: [{frame_id, bbox, ...}]}
        max_gap_frames:     Max frame gap to still consider as same robot.
        max_centre_dist_px: Max centre distance (pixels) between end-A and start-B.

    Returns:
        Deduplicated {track_id: [{frame_id, bbox, ...}]}
    """
    def _cx_cy(bbox):
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def _dist(a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    # Sort each track by frame_id; build summary (start, end, last_pos, first_pos)
    sorted_tracks: dict[int, list[dict]] = {
        tid: sorted(entries, key=lambda e: e["frame_id"])
        for tid, entries in robot_tracks.items()
    }
    summaries: dict[int, dict] = {}
    for tid, entries in sorted_tracks.items():
        summaries[tid] = {
            "start":      entries[0]["frame_id"],
            "end":        entries[-1]["frame_id"],
            "first_pos":  _cx_cy(entries[0]["bbox"]),
            "last_pos":   _cx_cy(entries[-1]["bbox"]),
            "frame_set":  {e["frame_id"] for e in entries},
        }

    # Union-find for merging
    parent: dict[int, int] = {tid: tid for tid in sorted_tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            keep = min(ra, rb)   # lower ID wins
            drop = max(ra, rb)
            parent[drop] = keep

    tids = sorted(sorted_tracks.keys())
    for i, tid_a in enumerate(tids):
        sa = summaries[tid_a]
        for tid_b in tids[i+1:]:
            sb = summaries[tid_b]
            # Must not overlap in time
            if sa["frame_set"] & sb["frame_set"]:
                continue
            # Identify which comes first
            if sa["end"] < sb["start"]:
                gap       = sb["start"] - sa["end"]
                end_pos   = sa["last_pos"]
                start_pos = sb["first_pos"]
            elif sb["end"] < sa["start"]:
                gap       = sa["start"] - sb["end"]
                end_pos   = sb["last_pos"]
                start_pos = sa["first_pos"]
            else:
                continue   # overlapping (shouldn't happen given check above)
            if gap > max_gap_frames:
                continue
            if _dist(end_pos, start_pos) > max_centre_dist_px:
                continue
            union(tid_a, tid_b)

    # Build merged tracks grouped by root ID
    merged: dict[int, list[dict]] = {}
    for tid, entries in sorted_tracks.items():
        root = find(tid)
        merged.setdefault(root, []).extend(entries)

    # Sort each merged track by frame_id
    for tid in merged:
        merged[tid].sort(key=lambda e: e["frame_id"])

    n_merged = len(robot_tracks) - len(merged)
    if n_merged > 0:
        print(f"  [Track] Dedup: merged {n_merged} fragment tracks "
              f"({len(robot_tracks)} -> {len(merged)})")
    return merged


# ---- Robot identity map -------------------------------------------------------

def build_robot_identity_map(
    robot_tracks:            dict[int, list[dict]],
    read_bumper_number_fn:   Callable[[np.ndarray, int], str],
    video_path:              str | Path,
    frame_sample:            int = 600,
) -> dict[int, dict]:
    """
    Map each robot track_id to a team number by running bumper OCR.

    Improvements over naive first-N-frames approach:
      • Scores every tracked frame by (bbox_area × sharpness); picks the top
        ``crops_per_track`` frames per track so OCR sees the clearest images.
      • Sequential video pass instead of per-frame random seeks (much faster
        on H.264 video where random seeks decode from the prior keyframe).
      • Final fallback: if weighted voting is still inconclusive, returns the
        single most-seen reading rather than UNKNOWN.

    Args:
        robot_tracks:           {track_id: [{frame_id, bbox, ...}]}
        read_bumper_number_fn:  Callable(crop_array, track_id) -> str
        video_path:             Source video (to extract frame crops).
        frame_sample:           Max distinct frame IDs to fetch from video.

    Returns:
        {
          track_id: {
            team_number: str,
            confidence:  float,
            frames_confirmed: int,
          }
        }
    """
    import math as _math
    from collections import Counter

    CROPS_PER_TRACK = 40   # top N clearest frames per robot

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # ── Step 1: score every entry by bbox area (larger = more pixels = better OCR)
    # We don't know sharpness yet (need the actual pixels), so use area as proxy.
    # For each track pick top CROPS_PER_TRACK by area, spread across the match.
    wanted_fids: set[int] = set()
    per_track_best: dict[int, list[dict]] = {}

    for track_id, entries in robot_tracks.items():
        scored = []
        for e in entries:
            b = e["bbox"]
            area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            scored.append((area, e))
        # Sort descending by area, take top CROPS_PER_TRACK
        scored.sort(key=lambda x: x[0], reverse=True)
        best = [e for _, e in scored[:CROPS_PER_TRACK]]
        per_track_best[track_id] = best
        wanted_fids.update(e["frame_id"] for e in best)

    # Cap total frames fetched to frame_sample (avoid OOM on long videos)
    wanted_sorted = sorted(wanted_fids)
    if len(wanted_sorted) > frame_sample:
        # Evenly sub-sample so we still cover the full match timeline
        step = len(wanted_sorted) / frame_sample
        wanted_sorted = [wanted_sorted[int(i * step)] for i in range(frame_sample)]
    wanted_set = set(wanted_sorted)

    # ── Step 2: single sequential video pass to collect frames
    # This is O(total_frames) decode but only one pass — far faster than
    # N random seeks on H.264 which each decode from the prior keyframe.
    frame_cache: dict[int, np.ndarray] = {}
    fid_iter = iter(sorted(wanted_set))
    target = next(fid_iter, None)
    current_fid = 0

    while target is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_cache[target] = frame
        target = next(fid_iter, None)

    cap.release()
    print(f"  [OCR] Fetched {len(frame_cache)} frames for robot identity OCR")

    # ── Step 3: run OCR on best crops for each track
    identity_map: dict[int, dict] = {}

    for track_id, best_entries in per_track_best.items():
        readings: list[str] = []

        for entry in best_entries:
            fid = entry["frame_id"]
            frame = frame_cache.get(fid)
            if frame is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in entry["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            result = read_bumper_number_fn(crop, track_id)
            if result and not str(result).startswith("UNKNOWN"):
                readings.append(str(result))

        if readings:
            most_common, count = Counter(readings).most_common(1)[0]
            conf = count / len(readings)
        else:
            # Soft fallback: look inside the OCR cache for any partial signal
            from detect import _ocr_cache as _oc
            cached = _oc.get(track_id, {})
            wc = cached.get("weighted_counts", {})
            if wc:
                most_common = max(wc, key=wc.get)
                total = sum(wc.values())
                conf  = wc[most_common] / total if total else 0.0
            else:
                most_common = f"UNKNOWN_{track_id}"
                conf = 0.0

        identity_map[track_id] = {
            "team_number":      most_common,
            "confidence":       round(conf, 3),
            "frames_confirmed": len(readings),
        }

    return identity_map


# ---- Alliance detection (color + position) -----------------------------------

def detect_alliances(
    robot_tracks: dict[int, list[dict]],
    video_path:   "str | Path",
    frame_sample: int = 300,
) -> dict[int, str]:
    """
    Determine alliance (red/blue) for each robot track using two signals:
      1. Bumper HSV color  — red bumpers → red alliance, blue → blue alliance
      2. Field position    — robots starting on left half = one alliance,
                             right half = other (confirmed by color)

    Args:
        robot_tracks:  {track_id: [track_dicts]} from run_robot_tracker()
        video_path:    Match video (to read frames for color sampling)
        frame_sample:  Only examine first N frames

    Returns:
        {track_id: "red" | "blue" | "unknown"}
    """
    alliance_map: dict[int, str] = {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return alliance_map

    # Sample up to 20 frames per track from the first frame_sample frames
    for track_id, entries in robot_tracks.items():
        red_votes = 0
        blue_votes = 0
        sampled = 0

        for entry in entries:
            if entry["frame_id"] >= frame_sample:
                break
            if sampled >= 20:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, entry["frame_id"])
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in entry["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Analyze bottom 30% of crop (bumper location)
            h = crop.shape[0]
            bumper = crop[int(h * 0.70):, :]
            if bumper.size == 0:
                bumper = crop

            hsv = cv2.cvtColor(bumper, cv2.COLOR_BGR2HSV)

            # Red: hue 0-10 or 160-180
            red_lo1 = cv2.inRange(hsv, np.array([0,   100, 80]),
                                        np.array([10,  255, 255]))
            red_lo2 = cv2.inRange(hsv, np.array([160, 100, 80]),
                                        np.array([180, 255, 255]))
            red_mask  = cv2.bitwise_or(red_lo1, red_lo2)
            blue_mask = cv2.inRange(hsv, np.array([100, 100, 80]),
                                         np.array([140, 255, 255]))

            red_px  = int(np.sum(red_mask  > 0))
            blue_px = int(np.sum(blue_mask > 0))

            if red_px > blue_px * 1.5:
                red_votes += 1
            elif blue_px > red_px * 1.5:
                blue_votes += 1

            sampled += 1

        if red_votes > blue_votes:
            alliance_map[track_id] = "red"
        elif blue_votes > red_votes:
            alliance_map[track_id] = "blue"
        else:
            # Fall back: left-half of frame = red side (typical broadcast layout)
            first = next((e for e in entries if e["frame_id"] < frame_sample), None)
            if first:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                frame_w = frame.shape[1] if (ret and frame is not None) else 1280
                cx = (first["bbox"][0] + first["bbox"][2]) / 2
                alliance_map[track_id] = "red" if cx < frame_w / 2 else "blue"
            else:
                alliance_map[track_id] = "unknown"

    cap.release()

    # Print summary
    for tid, alliance in sorted(alliance_map.items()):
        print(f"  [Alliance] Track {tid} -> {alliance.upper()}")

    return alliance_map


# ---- Calibration (per-match robot ID) ---------------------------------------

def calibrate_robot_identities(
    video_path:     str | Path,
    all_frame_detections: list[dict],
    project_root:   str | Path = ".",
    frame_sample:   int   = 300,
    confirm:        bool  = True,
) -> dict[int, dict]:
    """
    Full per-match robot identity calibration pipeline.

    Steps:
      1. Run robot tracker on first frame_sample frames of detections
      2. Run bumper OCR on each robot crop (majority vote per track)
      3. Print detected robots clearly
      4. STOP and ask user to confirm or correct team numbers
      5. Save confirmed identity map to configs/match_identity.json

    Args:
        video_path:              Source match video.
        all_frame_detections:    All per-frame detection dicts.
        project_root:            Project root.
        frame_sample:            Frames to analyse.
        confirm:                 If True, pause for user confirmation.

    Returns:
        {track_id: {team_number, confidence, frames_confirmed}}
    """
    from detect import read_bumper_number

    video_path = Path(video_path)
    root       = Path(project_root).resolve()

    print("\n  [Calibrate] Sampling first {} frames for robot ID...".format(frame_sample))

    # Limit to first frame_sample frames
    sample_dets = [f for f in all_frame_detections if f["frame_id"] < frame_sample]

    robot_tracks = run_robot_tracker(sample_dets)
    identity_map = build_robot_identity_map(
        robot_tracks, read_bumper_number, video_path, frame_sample
    )

    print("\n  Detected robots this match:")
    for tid, info in sorted(identity_map.items()):
        print(f"    Track {tid} -> Team {info['team_number']}"
              f"  (confidence: {info['confidence']*100:.0f}%,"
              f" seen in {info['frames_confirmed']} frames)")

    if confirm:
        # In GUI mode (no real stdin), skip interactive prompt — user corrects
        # via the Robot Assignment panel in the Analyze page instead.
        try:
            import sys
            if not sys.stdin or not sys.stdin.isatty():
                print("\n  [Calibrate] GUI mode - skipping interactive prompt.")
                print("  Use the Robot Assignment panel to assign team numbers.")
            else:
                print("\n  Do these team numbers look correct?")
                print("  Enter 'yes' to confirm, or 'correct [track] to [team]' to fix:")
                while True:
                    ans = input("  > ").strip().lower()
                    if ans in ("yes", "y"):
                        break
                    if ans.startswith("correct "):
                        parts = ans.split()
                        if len(parts) == 4 and parts[2] == "to":
                            tid_  = int(parts[1])
                            tnum  = parts[3]
                            if tid_ in identity_map:
                                identity_map[tid_]["team_number"]    = tnum
                                identity_map[tid_]["confidence"]     = 1.0
                                identity_map[tid_]["user_corrected"] = True
                                print(f"  Updated Track {tid_} -> Team {tnum}")
                            else:
                                print(f"  Track {tid_} not found.")
                        else:
                            print("  Format: correct [track_id] to [team_number]")
                    else:
                        print("  Please enter 'yes' or 'correct [track] to [team]'.")
        except Exception:
            print("\n  [Calibrate] Skipping interactive prompt (non-interactive mode).")

    # Determine alliance by bumper HSV colour + field position
    alliance_map = detect_alliances(robot_tracks, video_path, frame_sample)

    robots_json = []
    for tid, info in sorted(identity_map.items()):
        # Save first-frame centroid so next run can do spatial matching
        entries = robot_tracks.get(tid, [])
        if entries:
            first = min(entries, key=lambda e: e["frame_id"])
            b = first["bbox"]
            sx, sy = (b[0]+b[2])/2, (b[1]+b[3])/2
        else:
            sx, sy = 0.0, 0.0
        robots_json.append({
            "track_id":        tid,
            "team_number":     info["team_number"],
            "alliance":        alliance_map.get(tid, "unknown"),
            "confidence":      info["confidence"],
            "frames_confirmed": info["frames_confirmed"],
            "user_corrected":  info.get("user_corrected", False),
            "start_x":         round(sx, 1),
            "start_y":         round(sy, 1),
        })

    match_identity = {
        "video_file":          video_path.name,
        "calibration_frames":  frame_sample,
        "calibrated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "robots":              robots_json,
        "user_confirmed":      True,
    }

    out_path = root / "configs" / "match_identity.json"
    with open(out_path, "w") as f:
        json.dump(match_identity, f, indent=2)
    print(f"\n  [Calibrate] Identity map saved -> {out_path}")

    return identity_map
