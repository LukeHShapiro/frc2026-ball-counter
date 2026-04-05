# FRC 2026 Ball Counter — Claude Code Instructions (v2)
# Per-Robot Scoring Attribution System

## Project Goal
Build a Python-based video analysis tool that:
1. Detects and tracks game pieces (balls) from FRC 2026 match footage
2. Identifies each robot on the field individually by team number
3. Tracks ball possession per robot
4. Tracks ball trajectory from robot to scoring zone
5. Attributes each score to the correct robot — even when the robot is not
   visible at the moment of scoring
6. Outputs a per-robot score timeline with confidence levels

---

## Project Structure (Create This Exactly)

```
frc2026-ball-counter/
├── data/
│   ├── raw_frames/
│   ├── labeled/
│   │   ├── balls/           # Ball bounding box labels
│   │   ├── robots/          # Robot bounding box labels
│   │   └── bumpers/         # Bumper number crop labels (for OCR)
│   └── splits/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   ├── base/                # YOLOv8s base weights
│   ├── trained/
│   │   ├── ball_detector/   # Trained ball detection model
│   │   └── robot_detector/  # Trained robot detection model
├── src/
│   ├── ingest.py            # Video → frames
│   ├── train.py             # Train both detection models
│   ├── detect.py            # Run ball + robot detection
│   ├── track.py             # DeepSORT tracker for balls AND robots
│   ├── possession.py        # Ball-to-robot possession assignment
│   ├── trajectory.py        # Ball flight path prediction + scoring zone entry
│   ├── inference_engine.py  # Core attribution logic (who scored?)
│   ├── scoreboard.py        # OCR scoreboard reader (score validation)
│   ├── count.py             # Per-robot score aggregation
│   ├── export.py            # CSV/JSON/video output
│   └── ui.py                # Gradio UI
├── configs/
│   ├── dataset_balls.yaml
│   ├── dataset_robots.yaml
│   └── field_config.json    # Scoring zone coordinates, field layout
├── tests/
│   ├── test_possession.py
│   ├── test_trajectory.py
│   ├── test_attribution.py
│   └── test_count.py
├── requirements.txt
└── main.py
```

---

## Core System Architecture

### The Attribution Pipeline (Critical — Build This Exactly)

Every ball that scores must pass through this chain:

```
Frame Input
    │
    ▼
Ball Detector ──────────────────────────────► Ball Track ID assigned
    │                                               │
    ▼                                               ▼
Robot Detector ──────────────────────────► Robot Track ID + Team Number (OCR)
    │                                               │
    ▼                                               │
Possession Engine ◄─────────────────────────────────
  (Which robot is closest to / controlling the ball?)
    │
    ▼
Possession Log: {ball_track_id: robot_track_id, frame, confidence}
    │
    ▼
Trajectory Engine
  (Is this ball moving toward a scoring zone?)
  (Predict landing zone from velocity vector)
    │
    ▼
Scoring Event Detector
  (Did ball enter scoring zone?)
  (Did scoreboard change?)
    │
    ▼
Attribution Engine ◄──── Possession Log (last known possessor)
  (Who gets credit?)  ◄──── Trajectory prediction (who launched it?)
    │                 ◄──── Proximity log (who was nearest zone?)
    ▼
Per-Robot Score += 1  (with confidence score attached)
    │
    ▼
Output: Team 1234: 12pts | Team 5678: 8pts | Team 9999: 6pts
```

---

## Build Order — Follow Phases Exactly

---

### PHASE 1 — Environment Setup

- Create all directories listed above
- Create requirements.txt with:
  ultralytics, opencv-python, deep-sort-realtime, gradio, pandas, numpy,
  torch, torchvision, Pillow, PyYAML, easyocr, scipy, filterpy
- Write setup_check.py that:
  - Verifies all packages install correctly
  - Detects and reports CUDA availability
  - Downloads YOLOv8s base weights to models/base/
  - Prints PASS/FAIL for each check
- Do NOT proceed to Phase 2 until setup_check.py passes

⚠️ STOP — Print: "PHASE 1 COMPLETE. Run setup_check.py and confirm all PASS."

---

### PHASE 2 — Video Ingestion (src/ingest.py)

```python
extract_frames(video_path, output_dir, fps=10)
  # Accepts .mp4, .mov, .avi
  # Returns: total_frames, duration, resolution

filter_duplicates(frames_dir, similarity_threshold=0.95)
  # Perceptual hash deduplication
  # Target: 500-2000 frames for labeling

log_video_metadata(video_path)
  # Saves metadata.json to data/
```

⚠️ STOP — Print:
"PHASE 2 COMPLETE. Ready for INPUT CHECKPOINT #1 (match video)."

---

### PHASE 3 — Training Data Preparation

#### Two Separate Detection Models Required

**Model A: Ball Detector**
- Detects game pieces only
- Classes: defined by user at INPUT CHECKPOINT #4

**Model B: Robot Detector**
- Detects robots and reads team numbers from bumpers
- Single class: ["robot"]
- Team number is read via EasyOCR on the cropped robot bounding box
- Not a YOLO class — it is extracted by OCR after detection

#### Data Validation Functions

```python
validate_labels(labeled_dir, expected_classes)
  # Checks YOLOv8 format (.txt files with class x y w h)
  # Reports malformed labels

split_dataset(labeled_dir, model_name, train=0.8, val=0.1, test=0.1)
  # Splits separately for each model
  # Outputs to data/splits/{model_name}/

generate_dataset_yaml(class_names, model_name, output_path)
  # Writes configs/dataset_balls.yaml and configs/dataset_robots.yaml
```

⚠️ STOP — Print:
"PHASE 3 REQUIRES INPUT CHECKPOINTS #2, #3, #4. Waiting for user data."

---

### PHASE 4 — Model Training (src/train.py)

```python
train_model(dataset_yaml, base_weights, model_name, epochs=75, imgsz=640)
  # Train one model (call twice: balls, then robots)
  # Saves best weights to models/trained/{model_name}/best.pt
  # Reports mAP50, precision, recall

estimate_training_time()
  # Detect GPU/CPU → estimate time → confirm with user before starting

train_all()
  # Trains ball detector first, then robot detector
  # Reports accuracy of each before proceeding
```

Minimum acceptable mAP50 before proceeding:
- Ball detector:  >= 0.70
- Robot detector: >= 0.65
- If below: print warning + ask user to confirm retrain or continue anyway

⚠️ STOP — Print:
"PHASE 4 COMPLETE. Ball mAP: [X]. Robot mAP: [X]. Awaiting INPUT CHECKPOINT #5."

---

### PHASE 5 — Detection + Tracking (src/detect.py + src/track.py)

detect.py:
```python
run_ball_detection(frame, ball_weights, conf=0.4)
  # Returns: [{bbox, confidence, class_name}]

run_robot_detection(frame, robot_weights, conf=0.5)
  # Returns: [{bbox, confidence}]

read_bumper_number(robot_crop)
  # Runs EasyOCR on cropped robot image
  # Returns: team_number (str) or "UNKNOWN"
  # Caches result per track_id — do NOT re-OCR every frame
```

track.py:
```python
run_ball_tracker(ball_detections)
  # DeepSORT on balls
  # Returns: [{frame_id, track_id, bbox, class}]

run_robot_tracker(robot_detections)
  # DeepSORT on robots
  # Returns: [{frame_id, track_id, bbox}]

build_robot_identity_map(robot_tracks, read_bumper_number_fn)
  # Maps track_id → team_number with confidence
  # Handles robot disappearing + reappearing (re-ID correctly)
  # Returns: {track_id: {team_number: str, confidence: float, frames_confirmed: int}}
```

---

### PHASE 6 — Possession Engine (src/possession.py)

CRITICAL module. Build carefully and test thoroughly.

```python
POSSESSION_RULES = {
    "proximity_threshold_px": 80,    # Ball within N pixels of robot bbox edge = possibly possessed
    "possession_min_frames": 3,      # Must be near robot for N consecutive frames to confirm
    "release_threshold_px": 150,     # Ball this far from all robots = uncontrolled / in flight
    "velocity_handoff_threshold": 5  # Ball speed in px/frame below this = likely held (not flying)
}

assign_possession(ball_tracks, robot_tracks, frame_id)
  # For each ball track in this frame:
  #   1. Check proximity to each robot bounding box
  #   2. Check ball velocity (slow + near robot = possessed)
  #   3. Assign possession if both conditions met for min_frames
  # Returns: {ball_track_id: {robot_track_id, team_number, confidence, frame_id}}

build_possession_log(all_frames_possessions)
  # Full possession timeline:
  # {ball_track_id: [{frame_id, robot_track_id, team_number, confidence}]}

get_last_possessor(ball_track_id, before_frame, possession_log)
  # Returns the last robot confirmed holding this ball before a given frame
  # CRITICAL: used by attribution engine when robot is not visible at goal

detect_handoff(ball_track_id, possession_log, frame_window=15)
  # Detects if ball was passed from one robot to another
  # Returns: {from_robot, to_robot, handoff_frame} or None
```

---

### PHASE 7 — Trajectory Engine (src/trajectory.py)

```python
compute_ball_velocity(ball_track_id, ball_tracks, frame_window=5)
  # dx/dy per frame using recent track history
  # Returns: {vx, vy, speed}

predict_trajectory(ball_position, ball_velocity, frames_ahead=30)
  # Linear + simple parabolic prediction
  # Returns: list of predicted (x, y) positions

will_enter_zone(predicted_positions, zone_bbox)
  # Returns: {will_score: bool, predicted_frame: int, confidence: float}

detect_scoring_event(ball_track_id, ball_tracks, scoring_zones, possession_log)
  # Detects when a ball enters a scoring zone
  # Triggers the attribution engine
  # Returns: {
  #     event_frame: int,
  #     zone: str,
  #     ball_track_id: int,
  #     last_possessor: str (team_number),
  #     trajectory_origin_robot: str (team_number),
  #     confidence: float
  # }
```

---

### PHASE 8 — Scoreboard OCR (src/scoreboard.py)

```python
locate_scoreboard(frame, coords=None)
  # If coords provided (from INPUT CHECKPOINT #6): use those
  # Otherwise: attempt auto-detection via template matching
  # Returns: bbox of scoreboard region

read_score(frame, scoreboard_bbox)
  # EasyOCR on scoreboard region
  # Returns: {red_score: int, blue_score: int, confidence: float}

detect_score_change(score_history, current_score, frame_id)
  # Compares current to previous reading
  # Returns: {changed: bool, delta: int, alliance: str, frame_id: int}

validate_attribution(robot_scores, scoreboard_scores)
  # Checks: sum(robot scores per alliance) == scoreboard score
  # Returns: {
  #     red_attributed: int, red_scoreboard: int, red_gap: int,
  #     blue_attributed: int, blue_scoreboard: int, blue_gap: int
  # }
```

⚠️ STOP — Print: "PHASE 8 READY. Awaiting INPUT CHECKPOINT #6 (scoreboard location)."

---

### PHASE 9 — Attribution Engine (src/inference_engine.py)

The core intelligence. Build this carefully.

```python
ATTRIBUTION_PRIORITY = [
    "trajectory_origin",  # Ball path traced back to specific robot — HIGHEST CONFIDENCE
    "last_possessor",     # Last robot confirmed holding the ball — HIGH CONFIDENCE
    "proximity_to_zone",  # Robot nearest scoring zone at time of score — MEDIUM CONFIDENCE
    "alliance_only",      # Only alliance known, robot unknown — FLAG AS UNATTRIBUTED
]

attribute_score(scoring_event, possession_log, robot_tracks, ball_tracks)
  # Walks ATTRIBUTION_PRIORITY in order, stops at first confident result
  # Returns: {
  #     team_number: str,
  #     method: str,       # Which method succeeded
  #     confidence: float, # 0.0 to 1.0
  #     notes: str         # Human-readable explanation of reasoning
  # }
```

#### Handle These 5 Cases Explicitly

```python
# CASE 1: Robot visible at goal, trajectory + possession agree
#   → confidence: 0.95+

# CASE 2: Robot shot ball, then moved away before ball entered goal
#   → trajectory predicts correct zone + last_possessor confirms same robot
#   → confidence: 0.85+

# CASE 3: Ball came from off-screen (robot never visible in video)
#   → last_possessor from possession_log is best available signal
#   → confidence: 0.60–0.75

# CASE 4: Scoreboard changed but no ball entry detected
#   (occluded zone, missed detection, etc.)
#   → proximity_to_zone at time of scoreboard change
#   → confidence: 0.40–0.60
#   → Flag as "INFERRED — LOW CONFIDENCE" in output

# CASE 5: Two or more robots near zone simultaneously
#   → Use possession_log to break tie
#   → If still tied: flag as "AMBIGUOUS — MANUAL REVIEW"
#   → Do NOT silently guess

build_score_timeline(all_scoring_events)
  # [{frame_id, timestamp, team_number, method, confidence, zone, notes}]

compute_final_scores(score_timeline)
  # Returns:
  # {
  #   "1234": {score: 14, high_conf: 12, med_conf: 1, low_conf: 1, unattributed: 0},
  #   "5678": {score: 9,  high_conf: 7,  med_conf: 1, low_conf: 1, unattributed: 0},
  # }
```

---

### PHASE 10 — Score Aggregation (src/count.py)

```python
aggregate_scores(score_timeline, robot_identity_map)
  # Groups by team_number
  # Separates high/medium/low confidence attributions

generate_accuracy_report(final_scores, scoreboard_validation)
  # Computes: attribution rate %, unattributed points, confidence distribution
  # Flags any mismatch between robot sum and scoreboard total
  # Prints clear discrepancy report if gap > 0
```

---

### PHASE 11 — Export + UI (src/export.py + src/ui.py)

export.py:
```python
export_csv(final_scores, score_timeline, output_path)
  # One row per scoring event:
  # timestamp, team_number, method, confidence, zone, notes

export_json(full_results, output_path)

export_annotated_video(video_path, ball_tracks, robot_tracks,
                        possession_log, score_timeline, output_path)
  # Per frame draws:
  #   - Ball bboxes + track IDs
  #   - Robot bboxes + team numbers
  #   - Possession line: ball → owning robot (color-coded)
  #   - Running per-robot score overlay (top corner, all 6 robots)
  #   - Confidence badge on each score event (+1 HIGH / +1 LOW / +1 ?)
```

ui.py (Gradio):
- Video file upload (.mp4/.mov/.avi)
- Progress bar with current phase label
- Per-robot score table:
  | Team | Total | High Conf | Med Conf | Low Conf |
- Score timeline chart (line chart, per robot over match time)
- Flag viewer: lists all AMBIGUOUS and LOW CONFIDENCE events
  with frame preview + manual override dropdown
- Discrepancy panel: shows if robot scores don't add up to scoreboard
- Export: CSV, JSON, annotated video

---

### PHASE 12 — Testing (tests/)

```python
# test_possession.py
test_proximity_assignment()
test_velocity_filter_prevents_false_positive()
test_handoff_detection()
test_reacquisition_after_occlusion()

# test_trajectory.py
test_velocity_computation()
test_zone_entry_prediction()
test_offscreen_trajectory_case()

# test_attribution.py
test_case_1_visible_robot_at_goal()
test_case_2_robot_moved_before_score()
test_case_3_robot_never_visible()
test_case_4_scoreboard_only_change()
test_case_5_ambiguous_dual_robot()

# test_count.py
test_alliance_sum_matches_scoreboard()
test_confidence_bucketing()
test_unattributed_flagging()
test_discrepancy_report_triggers()
```

⚠️ STOP — Print:
"PHASE 12 COMPLETE. All tests passed. System ready for full match analysis."

---

## INPUT CHECKPOINTS SUMMARY

Stop and wait at each. Do not proceed without explicit user confirmation.

| # | When | What User Must Provide |
|---|------|------------------------|
| 1 | After Phase 2 | Match video file path (.mp4/.mov/.avi) |
| 2 | Before Phase 3 | Labeled ball images (Roboflow YOLOv8 export → data/labeled/balls/) |
| 3 | Before Phase 3 | Labeled robot images (Roboflow YOLOv8 export → data/labeled/robots/) |
| 4 | Before Phase 3 | Ball class names (e.g. ["ball"] or ["red_ball","blue_ball"]) |
| 5 | After Phase 4 | Validation video with known per-robot scores |
| 6 | Phase 8 start | Scoreboard pixel coordinates OR confirm "auto-detect" |
| 7 | Phase 9 start | Scoring zone pixel coordinates OR confirm "auto-detect zones" |

---

## General Rules

- Python 3.10+
- All paths use pathlib.Path
- Every function has a docstring: args, returns, which checkpoint it depends on
- Print clear progress at every major step
- Never silently skip a checkpoint or a confidence check
- Default model: YOLOv8s for both detectors
- All configs loaded from files, never hardcoded
- Confidence score is MANDATORY on every attribution output
- If confidence < 0.50: flag event in UI for manual review — never silently guess
- After every match: run validate_attribution() and print discrepancy report
- If robot score sum != scoreboard: list every unattributed point explicitly

---

## ARCHITECTURAL DECISION — Robot Identification (Locked)

### Method: Generic Detection + Per-Match OCR

Robot lineups change every match. The system MUST NOT use team numbers as
YOLO classes. The correct architecture is:

  YOLO detects: "robot" (one class, any robot)
  OCR reads:    team number from bumper crop each match

### Per-Match Calibration (REQUIRED at start of every analysis run)

Before any possession or attribution logic runs, execute this sequence:

```python
calibrate_robot_identities(video_path, robot_weights, frame_sample=300)
  # 1. Sample first 300 frames of the match video
  # 2. Run robot detector on each frame
  # 3. Run EasyOCR on every robot crop found
  # 4. Aggregate OCR readings per track_id
  #    (majority vote across frames to get confident team number)
  # 5. Build identity map: {track_id: team_number}
  # 6. PRINT the result clearly:
  #      "Detected robots this match:"
  #      "  Track 1 → Team 1234  (confidence: 94%, seen in 187 frames)"
  #      "  Track 2 → Team 5678  (confidence: 88%, seen in 201 frames)"
  #      "  Track 3 → Team 9999  (confidence: 91%, seen in 193 frames)"
  #      "  Track 4 → Team 2468  (confidence: 86%, seen in 178 frames)"
  #      "  Track 5 → Team 1357  (confidence: 79%, seen in 164 frames)"
  #      "  Track 6 → Team 3690  (confidence: 83%, seen in 155 frames)"
  # 7. STOP and ask user to confirm or correct before proceeding
  #    "Do these team numbers look correct? (yes / correct [track] to [team])"
  # 8. Save confirmed identity map to configs/match_identity.json
  # 9. Only then proceed to possession + attribution analysis
```

### OCR Reliability Rules

```python
OCR_RULES = {
    "min_frames_for_confidence": 10,  # Must see number in 10+ frames to trust it
    "majority_vote_threshold": 0.70,  # 70% of readings must agree
    "fallback_if_unknown": "UNKNOWN_[track_id]",  # Never silently drop a robot
    "re_ocr_on_reappearance": True,   # If robot disappears 60+ frames, re-confirm on return
    "blur_rejection_threshold": 100,  # Skip crops with Laplacian variance below this (too blurry)
}
```

### Handling Unknown Team Numbers

If OCR cannot read a robot's bumper number confidently:
- Label it "UNKNOWN_[track_id]" in all outputs
- Still track its possession and attribute scores to it
- Flag all its attributed scores as "TEAM NUMBER UNCONFIRMED"
- Show a blurry frame example in the UI for manual identification
- Allow user to type the correct team number in the UI override panel
  and retroactively re-attribute all its scores

### Match Identity File

Save to configs/match_identity.json after calibration:
```json
{
  "match_id": "user_provided_or_auto",
  "video_file": "match.mp4",
  "calibration_frames": 300,
  "robots": [
    {"track_id": 1, "team_number": "1234", "alliance": "red", "confidence": 0.94},
    {"track_id": 2, "team_number": "5678", "alliance": "red", "confidence": 0.88},
    {"track_id": 3, "team_number": "9999", "alliance": "red", "confidence": 0.91},
    {"track_id": 4, "team_number": "2468", "alliance": "blue", "confidence": 0.86},
    {"track_id": 5, "team_number": "1357", "alliance": "blue", "confidence": 0.79},
    {"track_id": 6, "team_number": "3690", "alliance": "blue", "confidence": 0.83}
  ],
  "user_confirmed": true
}
```

Alliance (red/blue) is determined by:
1. Bumper color detection (red vs blue bumpers via HSV color range on crop)
2. Field position at calibration time (red robots start on red side)
Both signals are combined. If they conflict, ask user to confirm.

---

## PHASE 13 — Driving Style Analysis (src/driving_analysis.py)

### Goal
Analyze the movement patterns of each robot throughout the match and classify
their driving style. This runs in parallel with scoring attribution — it uses
the same robot_tracks data already collected in Phase 5.

---

### Driving Style Classes

Four styles, with definitions Claude Code must implement exactly:

```
DEFENSIVE
  - Robot spends significant time positioned between opponents and scoring zones
  - Frequently mirrors/follows an opponent robot's movement
  - Low personal scoring rate relative to field time
  - High proximity events with opponent robots (shadowing, not colliding)

RECKLESS
  - High average velocity sustained over the match
  - Elevated collision rate (sudden velocity drops from contact)
  - Erratic heading changes (high angular velocity variance)
  - Frequently enters restricted or contested zones at speed

SMOOTH
  - Consistent velocity profile (low variance)
  - Clean arc trajectories (low angular jerk)
  - Low collision rate
  - Predictable, efficient paths between positions

DEFENCE_PROOF
  - Maintains scoring rate even when a defensive robot is shadowing it
  - Successfully escapes defensive contact (velocity recovers quickly after contact)
  - Uses varied paths — does not repeat the same route (low path repetition score)
  - Scores despite high opponent proximity events
```

A robot can receive a **primary** and **secondary** style label.
Example: `SMOOTH (primary) / DEFENCE_PROOF (secondary)`

---

### Metrics to Compute Per Robot Per Match

```python
DRIVING_METRICS = {
    # Velocity metrics
    "avg_velocity_px_per_frame":   float,  # Mean speed across all frames
    "velocity_variance":           float,  # Consistency (low = smooth)
    "max_velocity":                float,  # Peak speed
    "velocity_recovery_rate":      float,  # Speed restored after collision (px/frame/frame)

    # Collision metrics
    "collision_count":             int,    # Sudden velocity drops > threshold while near opponent
    "collision_rate_per_minute":   float,

    # Heading / trajectory metrics
    "angular_velocity_variance":   float,  # Heading change rate variance (low = smooth)
    "path_repetition_score":       float,  # 0.0–1.0 (1.0 = always same route)
    "arc_smoothness":              float,  # Curvature consistency

    # Defensive metrics
    "time_in_opponent_half_pct":   float,  # % of match spent in opponent alliance zone
    "shadowing_events":            int,    # Times robot mirrored opponent for 2+ seconds
    "avg_distance_to_nearest_opponent": float,

    # Defence-proof metrics
    "scoring_under_pressure_rate": float,  # Scores per minute when defender within 120px
    "escape_success_rate":         float,  # % of shadowing events robot broke free within 3s
    "path_variety_score":          float,  # Inverse of path_repetition_score
}
```

---

### Module Functions

```python
# src/driving_analysis.py

def compute_robot_velocity(robot_track_id, robot_tracks, frame_window=3)
  # Computes per-frame velocity and heading for a robot
  # Returns: [{frame_id, vx, vy, speed, heading_deg}]

def detect_collisions(robot_track_id, robot_tracks, all_robot_tracks,
                      velocity_drop_threshold=0.4, proximity_threshold_px=100)
  # A collision = sudden speed drop >= 40% while within 100px of another robot
  # Returns: [{frame_id, opponent_track_id, pre_velocity, post_velocity}]

def compute_path_repetition(robot_track_id, robot_tracks, grid_resolution=50)
  # Divides field into grid cells
  # Measures how often robot visits same cells in same order
  # Returns: score 0.0 (totally varied) to 1.0 (always same path)

def detect_shadowing_events(robot_track_id, robot_tracks, all_robot_tracks,
                             follow_duration_frames=20, proximity_threshold_px=120)
  # Detects when this robot mirrors an opponent's movement for N+ frames
  # Used for: classifying DEFENSIVE robots AND measuring pressure on others
  # Returns: [{start_frame, end_frame, target_robot_track_id, duration_frames}]

def compute_scoring_under_pressure(robot_track_id, score_timeline,
                                    shadowing_events, proximity_threshold_px=120)
  # Counts scores made by this robot while a defender was within proximity threshold
  # Cross-references with score_timeline from Phase 9
  # Returns: {scores_under_pressure: int, scores_total: int, pressure_rate: float}

def compute_all_metrics(robot_track_id, robot_tracks, all_robot_tracks,
                         score_timeline, alliance_zones)
  # Runs all metric functions above for one robot
  # Returns: full DRIVING_METRICS dict for that robot

def classify_driving_style(metrics)
  # Input: DRIVING_METRICS dict
  # Returns: {
  #     primary_style: str,       # "DEFENSIVE" | "RECKLESS" | "SMOOTH" | "DEFENCE_PROOF"
  #     secondary_style: str,     # Same options or None
  #     confidence: float,        # 0.0–1.0
  #     style_scores: {           # Raw score for each style (for UI display)
  #         "DEFENSIVE":      float,
  #         "RECKLESS":       float,
  #         "SMOOTH":         float,
  #         "DEFENCE_PROOF":  float
  #     },
  #     key_evidence: [str]       # 2–4 bullet points explaining why this style was chosen
  # }

def classify_all_robots(robot_identity_map, robot_tracks, all_robot_tracks,
                          score_timeline, alliance_zones)
  # Runs classify_driving_style for every robot in the match
  # Returns: {team_number: driving_style_result}

def generate_driving_report(all_classifications, all_metrics)
  # Returns structured report:
  # {
  #     team_number: {
  #         style:          str,
  #         secondary:      str or None,
  #         confidence:     float,
  #         style_scores:   dict,
  #         key_evidence:   [str],
  #         metrics:        DRIVING_METRICS dict
  #     }
  # }
```

---

### Classification Thresholds (Tunable in field_config.json)

```json
"driving_classification": {
    "defensive": {
        "min_time_opponent_half_pct": 0.30,
        "min_shadowing_events": 3,
        "max_personal_score_rate": 0.5
    },
    "reckless": {
        "min_avg_velocity_percentile": 0.75,
        "min_collision_rate_per_min": 1.5,
        "min_angular_variance_percentile": 0.70
    },
    "smooth": {
        "max_velocity_variance_percentile": 0.30,
        "max_collision_rate_per_min": 0.5,
        "max_angular_variance_percentile": 0.30
    },
    "defence_proof": {
        "min_scoring_under_pressure_rate": 0.60,
        "min_escape_success_rate": 0.50,
        "max_path_repetition_score": 0.40
    }
}
```

All thresholds loaded from `configs/field_config.json` — never hardcoded.

---

### Integration Points

- Runs AFTER Phase 5 (robot_tracks available) and Phase 9 (score_timeline available)
- Shares `robot_identity_map` from track.py
- Shares `score_timeline` from inference_engine.py
- Shares `alliance_zones` from field_config.json
- Does NOT depend on ball detection — pure robot movement analysis

---

### Add to Export (src/export.py)

```python
export_driving_report_csv(driving_report, output_path)
  # One row per robot:
  # team_number, primary_style, secondary_style, confidence,
  # avg_velocity, collision_count, shadowing_events,
  # scoring_under_pressure_rate, escape_success_rate, path_repetition_score

export_driving_report_json(driving_report, output_path)
```

---

### Add to Annotated Video (src/export.py)

In `export_annotated_video()`, add per-robot driving style badge:
- Small label under each robot's bounding box: e.g. `[SMOOTH]` or `[DEFENSIVE]`
- Color coded:
  - DEFENSIVE:     Orange
  - RECKLESS:      Red
  - SMOOTH:        Green
  - DEFENCE_PROOF: Blue

---

### Add to UI (src/ui.py)

New tab: **"Driving Analysis"**

- Per-robot driving style card:
  - Team number + primary/secondary style badge
  - Confidence bar
  - Style score breakdown (bar chart: Defensive / Reckless / Smooth / Defence-proof)
  - Key evidence bullets (why this style was assigned)
- Metrics table: all DRIVING_METRICS per robot
- Velocity profile chart: speed over match time, per robot (line chart, all 6 robots)
- Collision event timeline: marks on match timeline where collisions occurred
- Shadowing event viewer: click any shadowing event → jump to that frame in video

---

### Add to Tests (tests/test_driving.py)

```python
test_velocity_computation_accuracy()
test_collision_detection_no_false_positives()
test_shadowing_detection_min_duration()
test_path_repetition_identical_paths()
test_path_repetition_random_paths()
test_classification_defensive_profile()
test_classification_reckless_profile()
test_classification_smooth_profile()
test_classification_defence_proof_profile()
test_dual_style_assignment()
test_thresholds_load_from_config()
```

---

### Build Order for Phase 13

1. Add driving classification thresholds to `configs/field_config.json`
2. Build `src/driving_analysis.py` — all functions above
3. Add driving analysis call to `main.py` after Phase 9 completes
4. Update `src/export.py` with driving report exports
5. Update `src/export.py` annotated video to include style badges
6. Update `src/ui.py` with Driving Analysis tab
7. Build `tests/test_driving.py` and run all tests

⚠️ STOP after tests pass — Print:
"PHASE 13 COMPLETE. Driving analysis ready. All [N] tests passed."

---

## PHASE 14 — TBA Alliance Builder (src/tba_client.py + src/alliance_builder.py)

### Goal
Pull team and event data from The Blue Alliance API v3, combine it with
the per-robot metrics collected in Phases 9 and 13 (scoring attribution +
driving analysis), and recommend optimal alliance picks for a given event.

This runs AFTER match analysis is complete. It is a standalone module —
it does not affect video processing.

---

### ⚠️ INPUT CHECKPOINT #8
```
YOU MUST PROVIDE before Phase 14 runs:
- TBA API key (generate at https://www.thebluealliance.com/account)
  → Stored in configs/tba_config.json (NEVER hardcoded)
- Event key for the event you are scouting
  (format: "2026txhou" — year + event code, found in TBA URL)
- Your team number (the team doing the picking)
```

---

### Part A — TBA API Client (src/tba_client.py)

```python
BASE_URL = "https://www.thebluealliance.com/api/v3"

# All functions must:
# - Pass X-TBA-Auth-Key header on every request
# - Use ETag caching (If-None-Match header) to avoid redundant fetches
# - Handle 304 Not Modified responses gracefully
# - Raise clear errors on 401 (bad key) and 404 (bad event key)
# - Cache responses to data/tba_cache/{endpoint_hash}.json
#   so repeated runs don't hammer the API

def get_team_info(team_number)
  # GET /team/frc{team_number}
  # Returns: {nickname, city, state_prov, rookie_year, website}

def get_event_teams(event_key)
  # GET /event/{event_key}/teams
  # Returns: list of {team_number, nickname, city, state_prov}

def get_event_rankings(event_key)
  # GET /event/{event_key}/rankings
  # Returns: [{rank, team_number, wins, losses, ties, ranking_points,
  #            avg_match_points, avg_bonus_points}]

def get_event_oprs(event_key)
  # GET /event/{event_key}/oprs
  # Returns: {
  #   oprs:  {team_key: float},   # Offensive Power Rating
  #   dprs:  {team_key: float},   # Defensive Power Rating
  #   ccwms: {team_key: float}    # Calculated Contribution to Winning Margin
  # }

def get_event_matches(event_key)
  # GET /event/{event_key}/matches
  # Returns: full match list with alliance compositions + scores

def get_team_event_status(team_number, event_key)
  # GET /team/frc{team_number}/event/{event_key}/status
  # Returns: {qual_rank, qual_average, playoff_status, alliance}

def get_team_history(team_number, year=2026)
  # GET /team/frc{team_number}/events/{year}/statuses
  # Returns: performance across all events this season

def get_event_predictions(event_key)
  # GET /event/{event_key}/predictions
  # Returns: TBA's own predicted win probabilities (use as one input signal)
```

---

### Part B — Alliance Builder (src/alliance_builder.py)

#### Data Sources Merged Per Team

For every team at the event, the builder merges:

| Source | Data |
|--------|------|
| TBA API | OPR, DPR, CCWM, ranking points, win/loss record |
| TBA API | Historical event performance (season-wide) |
| Our video analysis | Per-robot score (Phases 9–10) |
| Our video analysis | Attribution confidence breakdown |
| Our video analysis | Driving style classification (Phase 13) |
| Our video analysis | Collision rate, velocity profile, defence-proof score |

If a team has not been analyzed in video yet, it uses TBA data only
and flags the entry as `"video_data": false`.

---

#### Team Score Computation

```python
ALLIANCE_WEIGHTS = {
    # TBA-sourced signals
    "opr":                       0.20,
    "ccwm":                      0.15,
    "ranking_points_avg":        0.10,
    "win_rate":                  0.05,

    # Video analysis signals (only if video_data=True)
    "video_score_rate":          0.20,  # Goals per match from our data
    "high_confidence_score_pct": 0.10,  # % of scores we're confident about
    "defence_proof_score":       0.10,  # From driving analysis
    "smooth_score":              0.05,  # Smooth driver = consistent partner
    "collision_rate_penalty":   -0.05,  # Reckless robots penalized

    # Compatibility signals (computed between robots)
    "style_complement_bonus":    0.10,  # Explained below
}
# Weights loaded from configs/tba_config.json — never hardcoded
```

#### Style Complement Bonus

Alliance performance benefits from role diversity. Apply bonuses:

```python
COMPLEMENT_MATRIX = {
    # (robot_A_style, robot_B_style): bonus
    ("SMOOTH",        "DEFENCE_PROOF"): +0.08,  # Ideal scoring duo
    ("DEFENSIVE",     "SMOOTH"):        +0.10,  # Classic 2-scorer + defender
    ("DEFENSIVE",     "DEFENCE_PROOF"): +0.06,  # Defender + hard-to-stop scorer
    ("SMOOTH",        "SMOOTH"):        +0.04,  # Two consistent scorers
    ("RECKLESS",      "DEFENSIVE"):     -0.05,  # Liability — reckless + defender clash
    ("RECKLESS",      "RECKLESS"):      -0.10,  # Two reckless = chaos, penalize
}
# Applied pairwise across the 3-robot alliance
```

---

#### Alliance Recommendation Functions

```python
def build_team_composite_scores(event_key, our_team_number,
                                  video_analysis_results, driving_results)
  # Merges TBA data + video data for every team at event
  # Returns: {team_number: composite_score_dict}

def recommend_picks(our_team_number, event_key,
                     team_composite_scores, strategy="balanced")
  # strategy options:
  #   "balanced"      — maximize overall alliance score
  #   "score_heavy"   — prioritize highest scoring pick 1 + pick 2
  #   "defensive"     — include at least one DEFENSIVE robot as pick 2
  #   "safe"          — prioritize low collision rate and high confidence data

  # Returns:
  # {
  #   pick_1: {
  #     team_number: str,
  #     composite_score: float,
  #     reasoning: [str],       # 3-5 bullet points explaining the pick
  #     data_confidence: str,   # "HIGH" / "MEDIUM" / "LOW" (how much video data we have)
  #     tba_opr: float,
  #     video_score_rate: float or None,
  #     driving_style: str or None
  #   },
  #   pick_2: { ... },
  #   projected_alliance_score: float,
  #   style_synergy: str,       # e.g. "SMOOTH + DEFENSIVE + DEFENCE_PROOF"
  #   warnings: [str]           # e.g. "Pick 1 has HIGH collision rate — monitor in playoffs"
  # }

def recommend_do_not_pick(team_composite_scores, reason_threshold=0.40)
  # Returns list of teams to avoid and why
  # [{team_number, reason, composite_score}]

def generate_pick_list(our_team_number, event_key,
                        team_composite_scores, top_n=10)
  # Full ranked pick list, not just top 2
  # Returns: [{rank, team_number, composite_score, style, reasoning, warnings}]

def compare_teams(team_a, team_b, team_composite_scores)
  # Side-by-side comparison of two teams
  # Returns: structured dict of all metrics for both teams

def simulate_alliance(team_list, team_composite_scores)
  # Given a list of 3 team numbers, compute projected performance
  # Returns: {
  #     projected_score: float,
  #     style_synergy: str,
  #     strengths: [str],
  #     weaknesses: [str],
  #     overall_rating: float
  # }
```

---

### Part C — Do Not Pick Analysis

```python
def flag_risky_teams(team_composite_scores, thresholds)
  # Flags teams with:
  #   - collision_rate > threshold (reckless, may damage alliance partners)
  #   - low data confidence + mediocre TBA data (unknown risk)
  #   - RECKLESS style + low CCWM (aggressive but not winning)
  #   - Significant gap between our video score and TBA OPR
  #     (inconsistent performance — unpredictable)
  # Returns: [{team_number, flags: [str], risk_level: "HIGH"/"MEDIUM"}]
```

---

### Add to Export (src/export.py)

```python
export_pick_list_csv(pick_list, output_path)
  # rank, team_number, composite_score, driving_style, opr, video_score_rate,
  # data_confidence, key_reasoning, warnings

export_pick_list_json(full_alliance_recommendation, output_path)

export_do_not_pick_csv(risky_teams, output_path)
```

---

### Add to UI (src/ui.py)

New tab: **"Alliance Builder"**

Layout:
- **Event key input** + **API key status indicator** (green = connected)
- **Your team number** input
- **Strategy selector**: Balanced / Score Heavy / Defensive / Safe
- **Pick List table** (ranked, all teams at event):
  | Rank | Team | Composite | Style | OPR | Video Score/Match | Confidence | Warnings |
- **Top 3 Recommendation card**:
  - Pick 1 + Pick 2 + projected alliance score
  - Style synergy badge
  - Per-pick reasoning bullets
  - Warning banners for risky picks
- **Do Not Pick list** with reason per team
- **Team Comparison tool**: select any 2 teams → side-by-side metrics
- **Alliance Simulator**: input any 3 team numbers → projected score + analysis
- **Data coverage indicator**: shows which teams have video analysis vs TBA-only
- Export buttons: Pick List CSV, Full Report JSON

---

### Config File (configs/tba_config.json)

```json
{
    "api_key": "YOUR_TBA_KEY_HERE",
    "event_key": "2026txhou",
    "our_team_number": "XXXX",
    "cache_ttl_seconds": 300,
    "strategy": "balanced",
    "alliance_weights": {
        "opr": 0.20,
        "ccwm": 0.15,
        "ranking_points_avg": 0.10,
        "win_rate": 0.05,
        "video_score_rate": 0.20,
        "high_confidence_score_pct": 0.10,
        "defence_proof_score": 0.10,
        "smooth_score": 0.05,
        "collision_rate_penalty": -0.05,
        "style_complement_bonus": 0.10
    },
    "do_not_pick_thresholds": {
        "min_composite_score": 0.40,
        "max_collision_rate_per_min": 2.5
    }
}
```

---

### Add to Tests (tests/test_alliance_builder.py)

```python
test_tba_client_auth_header_present()
test_tba_client_etag_caching()
test_tba_client_handles_304()
test_tba_client_raises_on_bad_key()
test_composite_score_weights_sum_to_1()
test_style_complement_matrix_applied()
test_pick_recommendation_excludes_our_team()
test_do_not_pick_flags_reckless_high_collision()
test_team_with_no_video_data_uses_tba_only()
test_alliance_simulator_three_teams()
test_pick_list_sorted_descending()
test_tba_config_loads_from_file()
```

---

### Build Order for Phase 14

1. Create `configs/tba_config.json` template (with placeholder key)
2. Build `src/tba_client.py` — all API functions with caching
3. Build `src/alliance_builder.py` — composite scoring + recommendation
4. Update `src/export.py` with pick list exports
5. Update `src/ui.py` with Alliance Builder tab
6. Build `tests/test_alliance_builder.py` and run all tests
7. Add Phase 14 call to `main.py` after Phase 13

⚠️ STOP after tests — Print:
"PHASE 14 COMPLETE. Alliance builder ready. All [N] tests passed."

⚠️ STOP before running — Print:
"INPUT CHECKPOINT #8: Provide your TBA API key and event key in configs/tba_config.json before running alliance builder."
