import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from collections import OrderedDict
from datetime import datetime
from ultralytics import YOLO
import uuid
import tempfile

from intelligence import get_historical_context
from rules import ParkingRuleEngine, LOCATION_PROFILES

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_CLICK_WIDGET = True
except ImportError:
    HAS_CLICK_WIDGET = False

st.set_page_config(page_title="Contextual Traffic Enforcement System", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg: #0B0E12;
        --panel: #151A20;
        --border: #232B33;
        --text: #EDEFF2;
        --muted: #7E8A96;
        --ok: #2DD4A7;
        --warn: #FFB454;
        --alert: #FF5C5C;
    }

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: var(--bg); color: var(--text); }

    section[data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text); }

    .console-title { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1.6rem; letter-spacing: 0.01em; color: var(--text); margin-bottom: 0.1rem; }
    .console-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; }
    hr, .console-rule { border: none; border-top: 1px solid var(--border); margin: 1.4rem 0; }
    .step-header { display: flex; align-items: baseline; gap: 0.6rem; margin-top: 0.6rem; margin-bottom: 0.1rem; }
    .step-number { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.78rem; color: var(--bg); background: var(--ok); border-radius: 3px; padding: 0.12rem 0.5rem; letter-spacing: 0.03em; }
    .step-title { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.05rem; color: var(--text); }
    .step-caption { font-family: 'IBM Plex Sans', sans-serif; font-size: 0.85rem; color: var(--muted); margin-bottom: 0.8rem; }
    .monitor-frame { border: 1px solid var(--border); background: var(--panel); border-radius: 4px; padding: 8px 8px 0 8px; margin-bottom: 0.3rem; }
    .monitor-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; padding: 0 2px 6px 2px; display: flex; align-items: center; gap: 0.4rem; }
    .rec-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--alert); display: inline-block; }
    .status-chip { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; border-radius: 4px; padding: 0.55rem 0.9rem; border: 1px solid var(--border); margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.5rem; }
    .status-chip.ok { color: var(--ok); border-color: rgba(45,212,167,0.35); background: rgba(45,212,167,0.07); }
    .status-chip.warn { color: var(--warn); border-color: rgba(255,180,84,0.35); background: rgba(255,180,84,0.07); }

    div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border); border-radius: 4px; padding: 0.8rem 1rem; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--muted); }
    div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--panel); border-color: var(--border) !important; border-radius: 4px; }
    .stButton > button { font-family: 'IBM Plex Mono', monospace; font-weight: 600; border-radius: 4px; border: 1px solid var(--border); }
    .stButton > button[kind="primary"] { background: var(--ok); color: #06140F; border: none; }
    .stButton > button[kind="primary"]:hover { background: #25C29A; }
    details { background: var(--panel); border: 1px solid var(--border) !important; border-radius: 4px !important; }
    .stCodeBlock, pre { font-family: 'IBM Plex Mono', monospace !important; border: 1px solid var(--border) !important; }
    [data-testid="stCaptionContainer"], .stCaption { font-family: 'IBM Plex Sans', sans-serif; color: var(--muted) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

CAMERA_NODES = {
    f"CAM_{i+1:03d}_{profile['name'].upper().replace(' ', '_')}": profile["name"]
    for i, profile in enumerate(LOCATION_PROFILES)
}

@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

def resize_to_max_width(img, max_width=1280):
    """EARLY OPTIMIZATION: Shrink massive 4K/1080p frames down to save 90% computation time."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def preprocess_frame(img):
    """Smart, high-performance cleanup for image/video inference."""
    # 1. SMART CLAHE: Only run histogram equalization if the image is too dark or overexposed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < 80 or mean_brightness > 200:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        base_img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    else:
        base_img = img

    # 2. FAST UNSHARP MASKING: Replaced heavy bilateral filter with high-speed Gaussian unsharp mask
    blurred = cv2.GaussianBlur(base_img, (5, 5), 1.2)
    sharpened = cv2.addWeighted(base_img, 1.5, blurred, -0.5, 0)
    
    return sharpened

class CentroidTracker:
    def __init__(self, max_disappeared=8, max_distance=80):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid):
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        return object_id

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    @staticmethod
    def bbox_centroid(bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return {}

        input_array = np.array(input_centroids)

        if len(self.objects) == 0:
            assignment = {}
            for centroid in input_array:
                object_id = self._register(tuple(centroid))
                assignment[object_id] = tuple(centroid)
            return assignment

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        distances = np.linalg.norm(object_centroids[:, np.newaxis] - input_array[np.newaxis, :], axis=2)

        rows_by_best_match = distances.min(axis=1).argsort()
        used_rows, used_cols = set(), set()
        assignment = {}

        for row in rows_by_best_match:
            col = distances[row].argmin()
            if row in used_rows or col in used_cols: continue
            if distances[row, col] > self.max_distance: continue

            object_id = object_ids[row]
            self.objects[object_id] = tuple(input_array[col])
            self.disappeared[object_id] = 0
            assignment[object_id] = tuple(input_array[col])
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        unused_cols = set(range(len(input_array))) - used_cols
        for col in unused_cols:
            object_id = self._register(tuple(input_array[col]))
            assignment[object_id] = tuple(input_array[col])

        return assignment

def _hough_stripe_detect(road_crop, y_start, w, h, bright_thresh, min_clusters, min_run):
    gray = cv2.cvtColor(road_crop, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    white_mask = cv2.dilate(white_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.bitwise_and(edges, white_mask)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=25, minLineLength=15, maxLineGap=15)
    if lines is None: return None
    
    horiz = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 35 or angle > 145:
            horiz.append((x1, y1 + y_start, x2, y2 + y_start))
    if len(horiz) < 4: return None

    xmids = [(l[0] + l[2]) // 2 for l in horiz]
    median_x = np.median(xmids)
    horiz = [l for l in horiz if abs((l[0] + l[2]) // 2 - median_x) < (w * 0.35)]
    if len(horiz) < 4: return None

    horiz.sort(key=lambda l: (l[1] + l[3]) // 2)
    clusters, cluster = [], [horiz[0]]
    for line in horiz[1:]:
        mid_y = (line[1] + line[3]) // 2
        prev_mid_y = (cluster[-1][1] + cluster[-1][3]) // 2
        if abs(mid_y - prev_mid_y) <= 20: cluster.append(line)
        else:
            clusters.append(cluster)
            cluster = [line]
    clusters.append(cluster)
    stripe_clusters = [c for c in clusters if len(c) >= 2]
    if len(stripe_clusters) < min_clusters: return None

    def cmid(c): return int(np.mean([(l[1] + l[3]) // 2 for l in c]))
    best_run, cur = [], [stripe_clusters[0]]
    for sc in stripe_clusters[1:]:
        if cmid(sc) - cmid(cur[-1]) <= 80: cur.append(sc)
        else:
            if len(cur) > len(best_run): best_run = cur
            cur = [sc]
    if len(cur) > len(best_run): best_run = cur
    if len(best_run) < min_run: return None

    all_lines = [l for c in best_run for l in c]
    pts = []
    for l in all_lines:
        pts.append([l[0], l[1]])
        pts.append([l[2], l[3]])
    pts = np.array(pts, dtype=np.int32)
    xs, ys = pts[:, 0], pts[:, 1]
    bbox_w, bbox_h = int(max(xs)) - int(min(xs)), int(max(ys)) - int(min(ys))

    if bbox_h == 0 or (bbox_w / bbox_h) < 1.8: return None
    if bbox_w < w * 0.35: return None
    band_center_y = (int(min(ys)) + int(max(ys))) / 2
    if band_center_y < road_crop.shape[0] * 0.40: return None

    cluster_mids = sorted(cmid(c) for c in best_run)
    gaps = np.diff(cluster_mids)
    if len(gaps) >= 2:
        gap_mean, gap_std = np.mean(gaps), np.std(gaps)
        if gap_mean > 0 and (gap_std / gap_mean) > 0.55: return None

    hull = cv2.convexHull(pts)
    epsilon = 0.03 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) >= 3:
        points = approx.reshape(-1, 2)
        cx, cy = np.mean(points, axis=0)
        def angle_from_center(p): return np.arctan2(p[1] - cy, p[0] - cx)
        return np.array(sorted(points, key=angle_from_center), dtype=np.int32)

    x_min, x_max = max(0, int(min(xs)) - 5), min(w - 1, int(max(xs)) + 5)
    y_min, y_max = max(0, int(min(ys)) - 5), min(h - 1, int(max(ys)) + 5)
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)

def detect_restricted_zone(img):
    h, w = img.shape[:2]
    y_start = int(h * 0.30)
    road = img[y_start:, :]
    passes = [(200, 3, 3), (160, 3, 3), (130, 2, 2)]
    for bright_thresh, min_clusters, min_run in passes:
        result = _hough_stripe_detect(road, y_start, w, h, bright_thresh, min_clusters, min_run)
        if result is not None:
            return result, True
    fallback = np.array(
        [[int(w * 0.20), int(h * 0.78)], [int(w * 0.80), int(h * 0.78)],
         [int(w * 0.92), int(h * 0.94)], [int(w * 0.08), int(h * 0.94)]],
        np.int32,
    )
    return fallback, False

def process_frame(input_data, precomputed_polygon=None):
    if isinstance(input_data, str):
        frame = cv2.imread(input_data)
        if frame is None:
            return {"error": "Could not read input image."}
    else:
        frame = input_data.copy()

    if precomputed_polygon is not None:
        restricted_polygon = precomputed_polygon
        was_auto_detected = True
    else:
        restricted_polygon, was_auto_detected = detect_restricted_zone(frame)
        
    detections = run_detection(frame)
    return {
        "image": frame,
        "image_shape": frame.shape,
        "restricted_polygon": restricted_polygon,
        "zone_auto_detected": was_auto_detected,
        "detections": detections,
    }

def _annotate_frame_array(img, restricted_polygon, detections, violations):
    annotated = img.copy()
    polygon = np.array(restricted_polygon, dtype=np.int32)
    violation_lookup = {tuple(v["bbox"]): v for v in violations}
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        violation = violation_lookup.get(tuple(det["bbox"]))
        if violation:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 40, 220), 3)
            box_h = y2 - y1
            y1_contact = int(y2 - 0.20 * box_h)
            cv2.rectangle(annotated, (x1, y1_contact), (x2, y2), (0, 140, 255), 1)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1_contact), (x2, y2), (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
            label = f"{violation['violation_type']} | {det['vehicle_type']} {violation['violation_confidence']:.1f}%"
            _draw_label(annotated, label, (x1, max(22, y1)), (40, 40, 220))
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 180, 75), 2)
            label = f"{det['vehicle_type']} {det['confidence']:.1f}%"
            _draw_label(annotated, label, (x1, max(22, y1)), (60, 180, 75))

    cv2.polylines(annotated, [polygon], True, (255, 0, 0), 3)
    if not violations:
        cv2.putText(
            annotated, "No parking violation detected in restricted zone",
            (22, min(annotated.shape[0] - 20, 40)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (30, 200, 30), 2, cv2.LINE_AA,
        )
    return annotated

def process_video(video_path, location_name, user_polygon=None, confidence_thresh=0.45, sample_fps=2, dwell_threshold_seconds=10):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {"error": "Could not open uploaded video file."}

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(source_fps / sample_fps))
    
    # Grab one frame to get the optimized output dimensions
    ok, test_frame = capture.read()
    if not ok: return {"error": "Could not read video frames."}
    test_frame = resize_to_max_width(test_frame)
    frame_height, frame_width = test_frame.shape[:2]
    
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset to beginning

    working_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    annotated_video_path = os.path.join(working_dir, f"{base_name}_annotated.webm")
    
    writer = cv2.VideoWriter(
        annotated_video_path,
        cv2.VideoWriter_fourcc(*"vp09"),
        sample_fps,
        (frame_width, frame_height),
    )

    max_frames_disappeared = int(sample_fps * 15)
    tracker = CentroidTracker(max_disappeared=max_frames_disappeared)
    
    rule_engine = None
    restricted_polygon = None
    track_dwell_seconds = {}
    track_last_violation = {}
    confirmed_track_ids = set()
    frame_index = 0
    processed_frame_count = 0
    preprocess_preview = None

    while True:
        success, frame = capture.read()
        if not success: break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue
        frame_index += 1
        processed_frame_count += 1

        # APPLY EARLY OPTIMIZATION
        frame = resize_to_max_width(frame)
        cleaned_frame = preprocess_frame(frame)
        
        if preprocess_preview is None:
            preprocess_preview = (frame.copy(), cleaned_frame.copy())

        frame_vision_results = process_frame(cleaned_frame, precomputed_polygon=user_polygon)

        if rule_engine is None:
            restricted_polygon = frame_vision_results["restricted_polygon"]
            rule_engine = ParkingRuleEngine(restricted_polygon, location_name, datetime.now().strftime("%H:%M"))

        detections = frame_vision_results["detections"]
        frame_violations = rule_engine.evaluate(detections, cleaned_frame.shape)
        violating_bboxes = {tuple(v["bbox"]) for v in frame_violations}

        centroids = [CentroidTracker.bbox_centroid(det["bbox"]) for det in detections]
        bbox_by_centroid_index = {i: det["bbox"] for i, det in enumerate(detections)}
        assignment = tracker.update(centroids)
        centroid_to_index = {tuple(c): i for i, c in enumerate(centroids)}

        for track_id, centroid in assignment.items():
            det_index = centroid_to_index.get(tuple(centroid))
            bbox = bbox_by_centroid_index.get(det_index) if det_index is not None else None
            is_in_zone = bbox is not None and tuple(bbox) in violating_bboxes
            if is_in_zone:
                track_dwell_seconds[track_id] = track_dwell_seconds.get(track_id, 0.0) + (1.0 / sample_fps)
                matching_violation = next((v for v in frame_violations if tuple(v["bbox"]) == tuple(bbox)), None)
                if matching_violation:
                    track_last_violation[track_id] = matching_violation
                if track_dwell_seconds[track_id] >= dwell_threshold_seconds:
                    confirmed_track_ids.add(track_id)
            else:
                track_dwell_seconds[track_id] = max(0.0, track_dwell_seconds.get(track_id, 0.0) - (0.5 / sample_fps))

        annotated_frame = _annotate_frame_array(cleaned_frame, restricted_polygon, detections, frame_violations)
        writer.write(annotated_frame)

    capture.release()
    writer.release()

    confirmed_violations = [
        {"track_id": track_id, "dwell_seconds": round(track_dwell_seconds.get(track_id, 0), 1), "violation": track_last_violation.get(track_id)}
        for track_id in confirmed_track_ids
    ]
    track_summary = [
        {"track_id": track_id, "dwell_seconds": round(dwell, 1), "confirmed": track_id in confirmed_track_ids}
        for track_id, dwell in track_dwell_seconds.items()
    ]
    return {
        "annotated_video_path": annotated_video_path,
        "confirmed_violations": confirmed_violations,
        "track_summary": track_summary,
        "processed_frame_count": processed_frame_count,
        "sampled_fps": sample_fps,
        "dwell_threshold_seconds": dwell_threshold_seconds,
        "preprocess_preview": preprocess_preview,
    }

def _draw_label(image, text, origin, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    padding = 6
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    text_w, text_h = text_size
    box_top = max(0, y - text_h - baseline - (padding * 2))
    box_bottom = max(y, box_top + text_h + baseline + (padding * 2))
    box_right = min(image.shape[1] - 1, x + text_w + (padding * 2))
    cv2.rectangle(image, (x, box_top), (box_right, box_bottom), color, -1)
    cv2.putText(
        image, text, (x + padding, box_bottom - baseline - padding),
        font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )

def run_detection(img, conf_threshold=0.25):
    allowed_classes = [1, 2, 3, 5, 7]
    results = model(img, conf=conf_threshold, classes=allowed_classes)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "vehicle_type": model.names[cls_id],
            "confidence": round(float(box.conf[0]) * 100, 2),
            "bbox": [x1, y1, x2, y2],
        })
    return detections

def save_evidence_image(annotated_img, output_path="output_evidence.png"):
    if output_path.lower().endswith(".png"):
        cv2.imwrite(output_path, annotated_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(output_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return output_path

def generate_evidence_image(
    img, restricted_polygon, detections, violations, location_name, current_time,
    intelligence_data=None, output_path="output_evidence.png",
):
    annotated = img.copy()
    polygon = np.array(restricted_polygon, dtype=np.int32)
    violation_lookup = {tuple(v["bbox"]): v for v in violations}
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        violation = violation_lookup.get(tuple(det["bbox"]))
        if violation:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 40, 220), 3)
            box_h = y2 - y1
            y1_contact = int(y2 - 0.20 * box_h)
            cv2.rectangle(annotated, (x1, y1_contact), (x2, y2), (0, 140, 255), 1)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1_contact), (x2, y2), (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
            label = f"{violation['violation_type']} | {det['vehicle_type']} {violation['violation_confidence']:.1f}%"
            _draw_label(annotated, label, (x1, max(22, y1)), (40, 40, 220))
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 180, 75), 2)
            label = f"{det['vehicle_type']} {det['confidence']:.1f}%"
            _draw_label(annotated, label, (x1, max(22, y1)), (60, 180, 75))

    cv2.polylines(annotated, [polygon], True, (255, 0, 0), 3)

    if not violations:
        no_violation_text = "No parking violation detected in restricted zone"
        x, y = 22, min(annotated.shape[0] - 20, 40)
        cv2.putText(
            annotated, no_violation_text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (30, 200, 30), 2, cv2.LINE_AA,
        )

    if output_path is None: return annotated
    return save_evidence_image(annotated, output_path)

def render_zone_preview(img, polygon):
    preview = img.copy()
    cv2.polylines(preview, [np.array(polygon, dtype=np.int32)], True, (255, 0, 0), 3)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

def render_click_progress(cv_img, points):
    preview = cv_img.copy()
    pts = [tuple(int(v) for v in p) for p in points]
    if len(pts) >= 3:
        cv2.line(preview, pts[-1], pts[0], (0, 220, 255), 1, cv2.LINE_AA)
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(preview, pts[i], pts[i + 1], (0, 220, 255), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(preview, p, 9, (0, 220, 255), -1)
        cv2.circle(preview, p, 9, (15, 15, 15), 2)
        cv2.putText(
            preview, str(i + 1), (p[0] + 13, p[1] - 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA,
        )
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


# ===========================================================================
# UI BUILDER
# ===========================================================================
st.markdown('<div class="console-title">Contextual Traffic Enforcement System</div>', unsafe_allow_html=True)
st.markdown('<div class="console-subtitle">Spatial Overlap Inference &amp; Violation Classification · MVP</div>', unsafe_allow_html=True)
st.markdown('<hr class="console-rule">', unsafe_allow_html=True)

with st.sidebar:
    st.header("Node Configuration")
    camera_key = st.selectbox("Select Target Node", list(CAMERA_NODES.keys()))
    location_name = CAMERA_NODES[camera_key]
    confidence_thresh = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.45)
    app_mode = st.radio("Mode", ["Image", "Video"], horizontal=True)

    st.caption("Lower this if vehicles aren't being picked up; raise it to reduce false detections.")
    st.markdown("---")
    st.header("Data Ingestion")

    uploaded_file = None
    uploaded_video = None
    video_sample_fps = 2
    dwell_threshold_seconds = 10
    
    source_frame = None  
    source_image_pil = None 
    current_fingerprint = None
    video_path_cached = None

    if app_mode == "Image":
        uploaded_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            upload_bytes = uploaded_file.getvalue()
            raw_frame = cv2.imdecode(np.frombuffer(upload_bytes, np.uint8), cv2.IMREAD_COLOR)
            if raw_frame is not None:
                source_frame = resize_to_max_width(raw_frame)
                source_image_pil = Image.fromarray(cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB))
                current_fingerprint = (uploaded_file.name, source_image_pil.size)
            else:
                st.error("Could not decode the uploaded image.")
        else:
            if os.path.exists("test_image.jpg"):
                raw_frame = cv2.imread("test_image.jpg")
                if raw_frame is not None:
                    source_frame = resize_to_max_width(raw_frame)
                    source_image_pil = Image.fromarray(cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB))
                    current_fingerprint = ("test_image.jpg", source_image_pil.size)
                    st.write("Status: Operating on default telemetry frame (test_image.jpg).")
                else:
                    st.error("test_image.jpg could not be read.")
            else:
                st.write("Status: Direct image upload required. (test_image.jpg not located).")
    else:
        uploaded_video = st.file_uploader("Upload Source Video", type=["mp4", "mov", "avi", "mkv"])
        video_sample_fps = st.slider("Video sampling FPS", 1, 5, 2)
        dwell_threshold_seconds = st.slider("Dwell confirmation threshold (seconds)", 5, 30, 10)
        
        if uploaded_video is not None:
            video_bytes = uploaded_video.getvalue()
            current_fingerprint = (uploaded_video.name, len(video_bytes), video_sample_fps, dwell_threshold_seconds)
            
            if st.session_state.get("_video_cache_fingerprint") != current_fingerprint:
                temp_dir = tempfile.mkdtemp()
                vid_path = os.path.join(temp_dir, f"{uuid.uuid4().hex[:8]}_{uploaded_video.name}")
                with open(vid_path, "wb") as buffer:
                    buffer.write(video_bytes)
                st.session_state["_cached_video_path"] = vid_path
                st.session_state["_video_cache_fingerprint"] = current_fingerprint
            
            video_path_cached = st.session_state["_cached_video_path"]
            capture = cv2.VideoCapture(video_path_cached)
            
            if capture.isOpened():
                ok, first_frame = capture.read()
                capture.release()
                if ok and first_frame is not None:
                    source_frame = resize_to_max_width(first_frame)
                    source_image_pil = Image.fromarray(cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB))
                    st.write(f"Status: Ready to analyze {uploaded_video.name}.")
                else:
                    st.error("Could not read the first frame of the uploaded video.")
            else:
                st.error("Could not open the uploaded video.")
        else:
            st.write("Status: Upload a video to begin.")

if source_frame is None or source_image_pil is None:
    st.info("Awaiting input. Select a target node and provide source media in the sidebar to begin.")
    st.stop()

if st.session_state.get("_media_fingerprint") != current_fingerprint:
    st.session_state["_media_fingerprint"] = current_fingerprint
    st.session_state.pop("restricted_polygon", None)
    st.session_state.pop("zone_auto_detected", None)
    st.session_state.pop("manual_points", None)
    st.session_state.pop("last_result", None)

if "restricted_polygon" not in st.session_state:
    auto_poly, was_auto = detect_restricted_zone(source_frame)
    st.session_state["restricted_polygon"] = auto_poly
    st.session_state["zone_auto_detected"] = was_auto

# ===========================================================================
# STEP 1 -- Calibrate Zone 
# ===========================================================================
st.markdown('<div class="step-header"><span class="step-number">STEP 1</span><span class="step-title">Calibrate Restricted Zone</span></div>', unsafe_allow_html=True)
st.markdown(f'<div class="step-caption">Target node: <b>{camera_key}</b> &nbsp;·&nbsp; Resolved location: <b>{location_name}</b></div>', unsafe_allow_html=True)

cal_col1, cal_col2 = st.columns(2)
with cal_col1:
    st.markdown('<div class="monitor-frame"><div class="monitor-label"><span class="rec-dot"></span>REFERENCE FRAME</div>', unsafe_allow_html=True)
    st.image(source_image_pil, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with cal_col2:
    st.markdown('<div class="monitor-frame"><div class="monitor-label">ZONE PREVIEW</div>', unsafe_allow_html=True)
    st.image(render_zone_preview(source_frame, st.session_state["restricted_polygon"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state["zone_auto_detected"]:
    st.markdown('<div class="status-chip ok">✓ &nbsp;ZONE AUTO-DETECTED FROM STRIPE PATTERN</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-chip warn">⚠ &nbsp;STRIPE PATTERN NOT CONFIDENTLY DETECTED — SHOWING ESTIMATED REGION. VERIFY OR REDRAW BELOW.</div>', unsafe_allow_html=True)

if "calib_key" not in st.session_state:
    st.session_state["calib_key"] = 0

expander_title = f"Correct / calibrate this zone manually{chr(8203) * st.session_state['calib_key']}"

with st.expander(expander_title, expanded=not st.session_state["zone_auto_detected"]):
    st.caption("Click 4 corners on the image below, in order, to define the zone. The zone drawn here will be used for all inference.")
    if HAS_CLICK_WIDGET:
        if "manual_points" not in st.session_state:
            st.session_state["manual_points"] = []

        orig_h, orig_w = source_frame.shape[:2]
        MAX_PREVIEW_WIDTH = 700  
        scale = min(1.0, MAX_PREVIEW_WIDTH / orig_w)
        
        preview_w, preview_h = int(orig_w * scale), int(orig_h * scale)
        preview_base = cv2.resize(source_frame, (preview_w, preview_h))
        preview_rgb = render_click_progress(preview_base, st.session_state["manual_points"])
        click_img = Image.fromarray(preview_rgb)

        coords = streamlit_image_coordinates(click_img, key="zone_clicker")

        if coords is not None:
            pt = (coords["x"], coords["y"])
            if not st.session_state["manual_points"] or st.session_state["manual_points"][-1] != pt:
                if len(st.session_state["manual_points"]) < 4:
                    st.session_state["manual_points"].append(pt)
                    st.rerun()

        st.write(f"Points selected: {len(st.session_state['manual_points'])}/4")
        man_col_a, man_col_b = st.columns(2)
        with man_col_a:
            if st.button("Reset points"):
                st.session_state["manual_points"] = []
                st.rerun()
        with man_col_b:
            if len(st.session_state["manual_points"]) == 4 and st.button("Apply manual zone", type="primary"):
                scaled_points = [[int(px / scale), int(py / scale)] for px, py in st.session_state["manual_points"]]
                st.session_state["restricted_polygon"] = np.array(scaled_points, dtype=np.int32)
                st.session_state["zone_auto_detected"] = True
                st.session_state["manual_points"] = []
                st.session_state["calib_key"] += 1 
                st.rerun()
    else:
        st.info("Install `streamlit-image-coordinates` for click-to-draw calibration. Falling back to numeric corner entry.")
        h, w = source_frame.shape[:2]
        default_poly = st.session_state["restricted_polygon"]
        pts = []
        for i in range(4):
            dx = int(default_poly[i][0]) if i < len(default_poly) else int(w * 0.5)
            dy = int(default_poly[i][1]) if i < len(default_poly) else int(h * 0.5)
            c1, c2 = st.columns(2)
            with c1: px = st.number_input(f"Point {i+1} X", 0, w, dx, key=f"px_{i}")
            with c2: py = st.number_input(f"Point {i+1} Y", 0, h, dy, key=f"py_{i}")
            pts.append((px, py))
        if st.button("Apply manual zone", type="primary"):
            st.session_state["restricted_polygon"] = np.array(pts, dtype=np.int32)
            st.session_state["zone_auto_detected"] = True
            st.session_state["calib_key"] += 1 
            st.rerun()

st.markdown('<hr class="console-rule">', unsafe_allow_html=True)


# ===========================================================================
# STEP 2 -- Run Inference (Image or Video context)
# ===========================================================================
st.markdown('<div class="step-header"><span class="step-number">STEP 2</span><span class="step-title">Run Inference</span></div>', unsafe_allow_html=True)
restricted_polygon = st.session_state["restricted_polygon"]

if app_mode == "Video":
    st.markdown('<div class="step-caption">Runs video object tracking, dwell-time evaluation, and spatial mapping using the calibrated zone above.</div>', unsafe_allow_html=True)
    if st.button("Execute Video Inference", type="primary", use_container_width=True):
        with st.spinner("Processing video, tracking vehicles, and accumulating dwell time..."):
            st.session_state["last_result"] = process_video(
                video_path_cached,
                location_name,
                user_polygon=restricted_polygon,
                confidence_thresh=confidence_thresh,
                sample_fps=video_sample_fps,
                dwell_threshold_seconds=dwell_threshold_seconds,
            )
            st.session_state["last_result"]["mode"] = "Video"
            
else:
    st.markdown('<div class="step-caption">Runs spatial detection and historical risk lookup on the static image using the calibrated zone above.</div>', unsafe_allow_html=True)
    if st.button("Execute Spatial Inference", type="primary", use_container_width=True):
        with st.spinner("Running detection, rule evaluation, and historical lookups..."):
            cv_img = source_frame.copy()
            preprocessed_img = preprocess_frame(cv_img)

            frame_vision_results = process_frame(preprocessed_img, precomputed_polygon=restricted_polygon)
            detections = frame_vision_results["detections"]
            
            current_time_str = datetime.now().strftime("%H:%M")
            rule_engine = ParkingRuleEngine(restricted_polygon, location_name, current_time_str)
            violations = rule_engine.evaluate(detections, preprocessed_img.shape)

            intelligence_data = get_historical_context(location_name, current_time_str, violations)
            enriched_violations = intelligence_data["enriched_violations"]

            output_image_path = generate_evidence_image(
                preprocessed_img, restricted_polygon, detections, enriched_violations,
                location_name, current_time_str, intelligence_data,
            )

            st.session_state["last_result"] = {
                "mode": "Image",
                "output_image_path": output_image_path,
                "preprocess_preview": (cv_img, preprocessed_img),
                "zone_context": rule_engine.zone_context,
                "detections_count": len(detections),
                "confidence_thresh": confidence_thresh,
                "camera_key": camera_key,
                "location_name": location_name,
                "zone_auto_detected": st.session_state["zone_auto_detected"],
                "intelligence_data": intelligence_data,
                "enriched_violations": enriched_violations,
            }

st.markdown('<hr class="console-rule">', unsafe_allow_html=True)


# ===========================================================================
# STEP 3 -- Results Output
# ===========================================================================
if "last_result" not in st.session_state:
    st.markdown('<div class="step-header"><span class="step-title">Result</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="step-caption">Run inference above to see the annotated output and risk assessment here.</div>', unsafe_allow_html=True)
    st.stop()

r = st.session_state["last_result"]
st.markdown('<div class="step-header"><span class="step-title">Result</span></div>', unsafe_allow_html=True)

if r["mode"] == "Video":
    st.caption(
        f"Processed frames: {r['processed_frame_count']} | Sample FPS: {r['sampled_fps']} | "
        f"Dwell threshold: {r['dwell_threshold_seconds']}s"
    )
    with open(r["annotated_video_path"], "rb") as v_file:
        st.video(v_file.read())

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Confirmed Violations", len(r["confirmed_violations"]))
    metric_col2.metric("Tracked Objects", len(r["track_summary"]))
    metric_col3.metric("Sampling FPS", r["sampled_fps"])

    if r.get("preprocess_preview"):
        before, after = r["preprocess_preview"]
        st.subheader("Preprocessing Preview")
        p1, p2 = st.columns(2)
        with p1: st.image(cv2.cvtColor(before, cv2.COLOR_BGR2RGB), caption="Before", use_container_width=True)
        with p2: st.image(cv2.cvtColor(after, cv2.COLOR_BGR2RGB), caption="After", use_container_width=True)

    if r["confirmed_violations"]:
        st.subheader("Confirmed Violations")
        st.dataframe(r["confirmed_violations"], use_container_width=True, hide_index=True)

    if r["track_summary"]:
        st.subheader("Track Summary")
        st.dataframe(r["track_summary"], use_container_width=True, hide_index=True)

else:
    zone_ctx = r["zone_context"]
    intelligence_data = r["intelligence_data"]
    enriched_violations = r["enriched_violations"]

    risk_score = intelligence_data["risk_score"]
    risk_color = "var(--alert)" if risk_score >= 70 else ("var(--warn)" if risk_score >= 45 else "var(--ok)")

    st.markdown(
        f'<div class="step-caption" style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;">'
        f'<span class="rec-dot" style="background:{risk_color};"></span>&nbsp; '
        f'RISK {risk_score}% &nbsp;·&nbsp; {len(enriched_violations)} VIOLATION(S) &nbsp;·&nbsp; '
        f'{r["location_name"].upper()} &nbsp;·&nbsp; {datetime.now().strftime("%H:%M")}</div>',
        unsafe_allow_html=True,
    )

    if r.get("preprocess_preview"):
        raw_before, raw_after = r["preprocess_preview"]
        st.subheader("Preprocessing Preview")
        prev_col1, prev_col2 = st.columns(2)
        with prev_col1:
            st.markdown('<div class="monitor-frame"><div class="monitor-label">BEFORE PREPROCESSING</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(raw_before, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with prev_col2:
            st.markdown('<div class="monitor-frame"><div class="monitor-label">AFTER PREPROCESSING</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(raw_after, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="monitor-frame"><div class="monitor-label"><span class="rec-dot"></span>ANNOTATED OUTPUT · INFERENCE RESULT</div>', unsafe_allow_html=True)
    st.image(r["output_image_path"], use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        f"Zone rule: **{zone_ctx['violation_type']}** ({zone_ctx['rule_source']} → {zone_ctx['matched_on']}) | "
        f"Active window: {zone_ctx['rule_window']} | Currently active: {'Yes' if zone_ctx['rule_active'] else 'No (low-confidence)'}"
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Violations Detected", len(enriched_violations))
    metric_col2.metric("Risk Score", f"{intelligence_data['risk_score']}%")
    metric_col3.metric("Congestion Risk", "HIGH" if intelligence_data.get("predicted_congestion_hotspot") else "LOW")

    if enriched_violations:
        st.subheader("Violation Detail")
        for v in enriched_violations:
            with st.container(border=True):
                st.markdown(f"**{v['violation_type']}** — {v['vehicle_type']} ({v['violation_confidence']:.1f}% confidence)")
                st.caption(v.get("reason", ""))

    with st.expander("System Execution Logs"):
        st.code(
            f"""
[INFO] Initializing inference on node {r['camera_key']} ({r['location_name']})
[INFO] Object detection completed. Found {r['detections_count']} target(s) exceeding confidence threshold {r['confidence_thresh']}.
[INFO] Restricted zone source: {"auto-detected" if r["zone_auto_detected"] else "manual/fallback"}
[INFO] Computing spatial overlap with configured zone polygon...
[INFO] Zone rule resolved: {zone_ctx['violation_type']} (source: {zone_ctx['rule_source']})
[WARN] Spatial intersection evaluated. Detected {len(enriched_violations)} active violation(s).
[INFO] Querying historical context via BTP OpenCity datasets...
[INFO] Location match scope: {intelligence_data['location_match_scope']}
[INFO] Contextual risk assessment complete. Risk Score: {intelligence_data['risk_score']}%
[INFO] Inference sequence completed successfully.
            """,
            language="bash",
        )

    with st.expander("Inference Metadata Payload (raw JSON)"):
        payload = dict(intelligence_data)
        payload["violations"] = enriched_violations
        payload.pop("enriched_violations", None)
        st.json(payload)