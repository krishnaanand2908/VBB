"""
cctv_ai_watch.py
- Uses webcam as CCTV
- Uses YOLOv8 (ultralytics) for object detection (people, custom classes)
- Uses MediaPipe Pose for pose/keypoint extraction and simple heuristics to detect fights/abuse
- Maintains rolling buffer; saves clip when suspicious event occurs and copies to highlights folder
"""

import cv2
import time
import json
import numpy as np
from collections import deque
from pathlib import Path
from datetime import datetime
import mediapipe as mp
import ultralytics

# Optional: ultralytics YOLOv8
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("ultralytics not found. Install with: pip install ultralytics")

# ---------- Configuration ----------
CAM_INDEX = 0  # default webcam
FPS = 20  # target FPS (used for buffering durations)
BUFFER_SECONDS = 10  # keep last N seconds
POST_EVENT_SECONDS = 8  # record N seconds after event triggered
OUT_DIR = Path("recordings")
HIGHLIGHT_DIR = Path("recordings/highlights")
META_LOG = OUT_DIR / "events.jsonl"
YOLO_WEIGHTS = None  # set path to custom weights if you have them, else None to use yolov8n.pt default
YOLO_CONF = 0.35
MIN_PERSON_CONF = 0.35

# Heuristic parameters for "fight" detection
PROXIMITY_PIXEL_THRESHOLD = 120  # distance (pixels) under which two people are "close"
RAISED_ARM_ANGLE_DEG = 40  # arm above horizontal threshold
MOVEMENT_VELOCITY_THRESHOLD = 5.0  # average landmark movement (pixels/frame) threshold

# Create folders
OUT_DIR.mkdir(parents=True, exist_ok=True)
HIGHLIGHT_DIR.mkdir(parents=True, exist_ok=True)

# Buffer size
BUFFER_SIZE = int(FPS * BUFFER_SECONDS)

# ---------- Helpers ----------
def now_str():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_clip(frames, fps, filename):
    """Save list of BGR frames to filename (mp4)"""
    if len(frames) == 0:
        return False
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))
    for f in frames:
        writer.write(f)
    writer.release()
    return True

def append_meta(event_dict):
    with open(META_LOG, "a") as f:
        f.write(json.dumps(event_dict) + "\n")

# ---------- Initialize models ----------
# MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# YOLO init (if available)
yolo_model = None
if YOLO is not None:
    if YOLO_WEIGHTS:
        yolo_model = YOLO(YOLO_WEIGHTS)
    else:
        # default detection model (tiny) - will download if needed
        yolo_model = YOLO("yolov8n.pt")
else:
    print("YOLO model not available. Object detection will be disabled.")

# ---------- Video capture ----------
cap = cv2.VideoCapture(CAM_INDEX)
# Try to set FPS (may or may not take effect)
cap.set(cv2.CAP_PROP_FPS, FPS)
actual_fps = cap.get(cv2.CAP_PROP_FPS) or FPS

frame_buffer = deque(maxlen=BUFFER_SIZE)
timestamp_buffer = deque(maxlen=BUFFER_SIZE)
landmark_history = deque(maxlen=BUFFER_SIZE)  # list of pose landmarks per frame

last_event_time = 0
is_recording_post = False
post_record_end_time = 0
post_record_frames = []

print("Starting CCTV AI watch. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        t0 = time.time()
        h, w = frame.shape[:2]
        display_frame = frame.copy()

        # Run YOLO inference (if available)
        yolo_results = []
        if yolo_model is not None:
            # Ultralytics .predict / .track returns a Results object; choose predict for simplicity
            # We run on a scaled-down image for speed
            try:
                # Resize for speed (optional)
                small = cv2.resize(frame, (640, int(640 * h / w)))
                res = yolo_model.predict(small, conf=YOLO_CONF, imgsz=640, verbose=False)
                # ulralytics returns list of results (one per image). We'll take first
                r = res[0]
                boxes = r.boxes
                for bx in boxes:
                    cls = int(bx.cls.cpu().numpy()) if hasattr(bx, "cls") else int(bx.data[5])
                    conf = float(bx.conf.cpu().numpy()) if hasattr(bx, "conf") else float(bx.data[4])
                    # convert box coords back to original frame size
                    xyxy = bx.xyxy[0].cpu().numpy()  # x1,y1,x2,y2 on small image
                    # scale coords
                    scale_x = w / small.shape[1]
                    scale_y = h / small.shape[0]
                    x1, y1, x2, y2 = xyxy[0]*scale_x, xyxy[1]*scale_y, xyxy[2]*scale_x, xyxy[3]*scale_y
                    yolo_results.append({
                        "class_id": cls,
                        "conf": conf,
                        "box": [int(x1), int(y1), int(x2), int(y2)]
                    })
                # draw boxes (optional)
                for det in yolo_results:
                    x1,y1,x2,y2 = det["box"]
                    cv2.rectangle(display_frame,(x1,y1),(x2,y2),(0,255,0),2)
            except Exception as e:
                # If ultralytics API differs by version, just skip detection this frame
                # Continue gracefully
                # print("YOLO inference skipped:", e)
                pass

        # Pose detection (MediaPipe)
        lm_list = None
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            lm_list = [(int(p.x * w), int(p.y * h), float(p.visibility)) for p in lm]
            # draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Save to buffers
        frame_buffer.append(frame.copy())
        timestamp_buffer.append(time.time())
        landmark_history.append(lm_list)

        # Simple heuristic for "fight/aggression" detection:
        # - look for two or more people close together (from YOLO person boxes)
        # - and at least one has raised arm angle or sudden movement
        trigger = False
        trigger_reasons = []

        # Derive person bounding boxes (YOLO class for 'person' is 0 in COCO)
        person_boxes = [d["box"] for d in yolo_results if d["class_id"] == 0 and d["conf"] >= MIN_PERSON_CONF]
        if len(person_boxes) >= 2:
            # compute pairwise distances between box centers
            centers = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in person_boxes]
            # If any pair is within pixel threshold, mark proximity
            close_pair = False
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dx = centers[i][0]-centers[j][0]
                    dy = centers[i][1]-centers[j][1]
                    dist = (dx*dx + dy*dy)**0.5
                    if dist < PROXIMITY_PIXEL_THRESHOLD:
                        close_pair = True
            if close_pair:
                trigger = True
                trigger_reasons.append("proximity_of_people")

        # Pose-based raised-arm detection and sudden motion
        if lm_list:
            # compute shoulder, elbow, wrist positions (MediaPipe landmarks indices)
            # right shoulder 12, right elbow 14, right wrist 16
            # left shoulder 11, left elbow 13, left wrist 15
            def angle_between(a,b,c):
                # angle at b between ba and bc
                ba = np.array(a) - np.array(b)
                bc = np.array(c) - np.array(b)
                cosang = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8)
                ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                return ang

            try:
                rs = lm_list[12]; re = lm_list[14]; rw = lm_list[16]
                ls = lm_list[11]; le = lm_list[13]; lw = lm_list[15]
                # angle at shoulder between horizontal and upper arm: approximate via shoulder-elbow-wrist
                r_ang = angle_between(rs[:2], re[:2], rw[:2])
                l_ang = angle_between(ls[:2], le[:2], lw[:2])
                # Simple check: small elbow angle might indicate raised arm (approximation)
                if r_ang < 160 or l_ang < 160:  # arm not straight down
                    # check movement velocity over last frames
                    # compute average movement of wrists over last 3 frames
                    movements = []
                    history = list(landmark_history)
                    if len(history) >= 3:
                        last = history[-1]
                        prev = history[-3]
                        if last and prev:
                            # take average of wrist deltas
                            lw_last = np.array(last[15][:2]); lw_prev = np.array(prev[15][:2])
                            rw_last = np.array(last[16][:2]); rw_prev = np.array(prev[16][:2])
                            m = (np.linalg.norm(lw_last-lw_prev) + np.linalg.norm(rw_last-rw_prev)) / 2.0
                            movements.append(m)
                    avg_move = np.mean(movements) if movements else 0.0
                    if avg_move > MOVEMENT_VELOCITY_THRESHOLD:
                        trigger = True
                        trigger_reasons.append("raised_arm_and_fast_motion")
            except Exception:
                # if any index missing, skip this heuristic
                pass

        # Also detect "fire" or "smoke" or custom classes via YOLO (if model trained)
        # We check yolo_results for classes that correspond to 'fire' or any custom label
        # (Requires custom YOLO model or dataset)
        suspicious_objects = []
        for d in yolo_results:
            # If you trained custom classes, map class_id to name here.
            # For COCO default, certain labels exist; you can print yolo_model.names to inspect.
            try:
                class_name = yolo_model.names[d["class_id"]]
            except Exception:
                class_name = str(d["class_id"])
            if class_name.lower() in ("fire","smoke","cigarette","handgun","knife","cracker","explosive"):
                suspicious_objects.append((class_name, d["conf"]))
        if suspicious_objects:
            trigger = True
            trigger_reasons.append("suspicious_objects:" + ",".join([x[0] for x in suspicious_objects]))

        # Event handling: if trigger and we are not already in post recording
        current_time = time.time()
        if trigger and (current_time - last_event_time) > (POST_EVENT_SECONDS + 1):
            last_event_time = current_time
            # collect frames from buffer (pre-event)
            pre_frames = list(frame_buffer)
            # start recording post-event frames
            is_recording_post = True
            post_record_end_time = current_time + POST_EVENT_SECONDS
            post_record_frames = []
            # Save an immediate copy of pre-event as well (we'll flush all later)
            print(f"[{now_str()}] Event triggered: {trigger_reasons}")

            # Prepare metadata
            event_meta = {
                "timestamp_utc": now_str(),
                "reasons": trigger_reasons,
                "num_pre_frames": len(pre_frames),
                "fps": actual_fps,
            }
            # Generate filename base
            fname_base = f"event_{now_str()}"
            event_meta["filename_base"] = fname_base
            append_meta({**event_meta, "status":"started"})

        # If in post-event recording window, collect frames
        if is_recording_post:
            post_record_frames.append(frame.copy())
            if time.time() >= post_record_end_time:
                # Save combined clip: pre + post
                all_frames = list(frame_buffer) + post_record_frames  # note: frame_buffer retains N frames; acceptable
                out_file = OUT_DIR / f"{fname_base}.mp4"
                saved = save_clip(all_frames, actual_fps or FPS, out_file)
                if saved:
                    # copy to highlights folder
                    highlight_file = HIGHLIGHT_DIR / f"{fname_base}_highlight.mp4"
                    import shutil
                    shutil.copy2(out_file, highlight_file)
                    append_meta({
                        "timestamp_utc": now_str(),
                        "status": "saved",
                        "file": str(out_file),
                        "highlight": str(highlight_file),
                        "reasons": trigger_reasons
                    })
                    print(f"Saved event clip: {out_file} and highlight: {highlight_file}")
                else:
                    append_meta({"timestamp_utc": now_str(), "status":"save_failed"})
                # reset post-event state
                is_recording_post = False
                post_record_frames = []

        # Show frame optionally
        cv2.imshow("CCTV AI Watch", display_frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q"):
            break

        # throttle to target FPS (light)
        t1 = time.time()
        elapsed = t1 - t0
        wait = max(0, (1.0 / (FPS)) - elapsed)
        if wait > 0:
            time.sleep(wait)

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Stopped.")

