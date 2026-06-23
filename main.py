from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Union
import os
import shutil
import uuid
import json

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from intelligence import get_historical_context
from rules import ParkingRuleEngine

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    return YOLO("yolov8s.pt")

model = load_model()

def resize_to_max_width(img: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def preprocess_frame(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < 80 or mean_brightness > 200:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        base_img = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
    else:
        base_img = img

    blurred = cv2.GaussianBlur(base_img, (5, 5), 1.2)
    sharpened = cv2.addWeighted(base_img, 1.5, blurred, -0.5, 0)
    return sharpened

def _draw_label(image: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
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
        image,
        text,
        (x + padding, box_bottom - baseline - padding),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

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
    xs = pts[:, 0]
    ys = pts[:, 1]
    bbox_w = int(max(xs)) - int(min(xs))
    bbox_h = int(max(ys)) - int(min(ys))

    if bbox_h == 0 or (bbox_w / bbox_h) < 1.8: return None
    if bbox_w < w * 0.35: return None
    band_center_y = (int(min(ys)) + int(max(ys))) / 2
    if band_center_y < road_crop.shape[0] * 0.40: return None

    cluster_mids = sorted(cmid(c) for c in best_run)
    gaps = np.diff(cluster_mids)
    if len(gaps) >= 2:
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)
        if gap_mean > 0 and (gap_std / gap_mean) > 0.55: return None

    hull = cv2.convexHull(pts)
    epsilon = 0.03 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) >= 3:
        points = approx.reshape(-1, 2)
        cx, cy = np.mean(points, axis=0)
        def angle_from_center(p): return np.arctan2(p[1] - cy, p[0] - cx)
        sorted_points = sorted(points, key=angle_from_center)
        return np.array(sorted_points, dtype=np.int32)

    x_min = max(0, int(min(xs)) - 5)
    x_max = min(w - 1, int(max(xs)) + 5)
    y_min = max(0, int(min(ys)) - 5)
    y_max = min(h - 1, int(max(ys)) + 5)
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)

def detect_restricted_zone(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    h, w = img.shape[:2]
    y_start = int(h * 0.30)
    road = img[y_start:, :]
    for bright_thresh, min_clusters, min_run in [(200, 3, 3), (160, 3, 3), (130, 2, 2)]:
        result = _hough_stripe_detect(road, y_start, w, h, bright_thresh, min_clusters, min_run)
        if result is not None:
            return result, True
    fallback = np.array(
        [
            [int(w * 0.20), int(h * 0.78)],
            [int(w * 0.80), int(h * 0.78)],
            [int(w * 0.92), int(h * 0.94)],
            [int(w * 0.08), int(h * 0.94)],
        ],
        np.int32,
    )
    return fallback, False

def run_detection(img: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
    allowed_classes = [1, 2, 3, 5, 7]
    results = model(img, conf=conf_threshold, classes=allowed_classes)[0]
    detections: List[Dict[str, Any]] = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "vehicle_type": model.names[cls_id],
            "confidence": round(float(box.conf[0]) * 100, 2),
            "bbox": [x1, y1, x2, y2],
        })
    return detections

def save_evidence_image(annotated_img: np.ndarray, output_path: str = "output_evidence.png") -> str:
    if output_path.lower().endswith(".png"):
        cv2.imwrite(output_path, annotated_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(output_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return output_path

def generate_evidence_image(
    img: np.ndarray, restricted_polygon, detections, violations, location_name, current_time,
    intelligence_data=None, output_path: str | None = "output_evidence.png",
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
        cv2.putText(
            annotated, "No parking violation detected in restricted zone",
            (22, min(annotated.shape[0] - 20, 40)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (30, 200, 30), 2, cv2.LINE_AA,
        )

    if output_path is None: return annotated
    return save_evidence_image(annotated, output_path)

def process_frame(input_data: Union[str, np.ndarray], precomputed_polygon=None) -> Dict[str, Any]:
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

@app.post("/analyze")
async def analyze_traffic(
    image: UploadFile = File(...),
    location: str = Form(...),
    timestamp: str = Form(None),
    custom_polygon: str = Form(None),
):
    unique_id = uuid.uuid4().hex[:8]
    sanitized_filename = f"{unique_id}_{image.filename}"
    file_path = os.path.join(UPLOAD_DIR, sanitized_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    if timestamp:
        current_time = datetime.fromisoformat(timestamp).strftime("%H:%M")
    else:
        current_time = datetime.now().strftime("%H:%M")
        
    parsed_polygon = None
    if custom_polygon:
        try:
            parsed_polygon = np.array(json.loads(custom_polygon), dtype=np.int32)
        except Exception:
            pass

    raw_frame = cv2.imread(file_path)
    if raw_frame is not None:
        raw_frame = resize_to_max_width(raw_frame)
        preprocessed_frame = preprocess_frame(raw_frame)
    else:
        return {"error": "Could not decode uploaded image."}

    vision_results = process_frame(preprocessed_frame, precomputed_polygon=parsed_polygon)
    if "error" in vision_results:
        return vision_results

    rule_engine = ParkingRuleEngine(
        vision_results["restricted_polygon"],
        location,
        current_time,
    )

    violations = rule_engine.evaluate(
        vision_results["detections"],
        vision_results["image_shape"],
    )

    intelligence_results = {}
    if len(violations) > 0:
        intelligence_results = get_historical_context(location, current_time, violations)
        violations = intelligence_results.get("enriched_violations", violations)
        intelligence_results.pop("enriched_violations", None)

    base_name, ext = os.path.splitext(sanitized_filename)
    evidence_filename = f"{base_name}_evidence{ext}"
    evidence_file_path = os.path.join(UPLOAD_DIR, evidence_filename)

    evidence_image = generate_evidence_image(
        preprocessed_frame,
        vision_results["restricted_polygon"],
        vision_results["detections"],
        violations,
        location,
        current_time,
        intelligence_results,
        output_path=evidence_file_path,
    )

    if evidence_image:
        vision_results["evidence_image"] = f"uploads/{evidence_filename}"

    return {
        "timestamp": current_time,
        "location": location,
        "violations": violations,
        "vision_data": vision_results,
        "intelligence_data": intelligence_results,
    }

VIDEO_SAMPLE_FPS = 2
DWELL_CONFIRMATION_SECONDS = 10

@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(...),
    location: str = Form(...),
    timestamp: str = Form(None),
    custom_polygon: str = Form(None),
):
    unique_id = uuid.uuid4().hex[:8]
    sanitized_filename = f"{unique_id}_{video.filename}"
    video_path = os.path.join(UPLOAD_DIR, sanitized_filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    if timestamp:
        current_time = datetime.fromisoformat(timestamp).strftime("%H:%M")
    else:
        current_time = datetime.now().strftime("%H:%M")
        
    parsed_polygon = None
    if custom_polygon:
        try:
            parsed_polygon = np.array(json.loads(custom_polygon), dtype=np.int32)
        except Exception:
            pass

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {"error": "Could not open uploaded video file."}

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(source_fps / VIDEO_SAMPLE_FPS))
    
    ok, test_frame = capture.read()
    if not ok:
        return {"error": "Could not extract frames from video."}
        
    test_frame = resize_to_max_width(test_frame)
    frame_height, frame_width = test_frame.shape[:2]
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    annotated_video_filename = f"{os.path.splitext(sanitized_filename)[0]}_annotated.webm"
    annotated_video_path = os.path.join(UPLOAD_DIR, annotated_video_filename)
    writer = cv2.VideoWriter(
        annotated_video_path,
        cv2.VideoWriter_fourcc(*"vp09"), 
        VIDEO_SAMPLE_FPS,
        (frame_width, frame_height),
    )

    max_frames_disappeared = int(VIDEO_SAMPLE_FPS * 15)
    tracker = CentroidTracker(max_disappeared=max_frames_disappeared)
    
    rule_engine = None
    restricted_polygon = None
    track_dwell_seconds = {}
    track_last_violation = {}
    confirmed_track_ids = set()

    frame_index = 0
    processed_frame_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % frame_interval != 0:
            frame_index += 1
            continue
        frame_index += 1
        processed_frame_count += 1

        frame = resize_to_max_width(frame)
        cleaned_frame = preprocess_frame(frame)
        
        frame_vision_results = process_frame(cleaned_frame, precomputed_polygon=parsed_polygon)
        if "error" in frame_vision_results:
            continue

        if rule_engine is None:
            restricted_polygon = frame_vision_results["restricted_polygon"]
            rule_engine = ParkingRuleEngine(restricted_polygon, location, current_time)

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
                track_dwell_seconds[track_id] = track_dwell_seconds.get(track_id, 0.0) + (1.0 / VIDEO_SAMPLE_FPS)
                matching_violation = next(
                    (v for v in frame_violations if tuple(v["bbox"]) == tuple(bbox)),
                    None,
                )
                if matching_violation:
                    track_last_violation[track_id] = matching_violation
                if track_dwell_seconds[track_id] >= DWELL_CONFIRMATION_SECONDS:
                    confirmed_track_ids.add(track_id)
            else:
                track_dwell_seconds[track_id] = max(
                    0.0,
                    track_dwell_seconds.get(track_id, 0.0) - (0.5 / VIDEO_SAMPLE_FPS),
                )

        annotated_frame = generate_evidence_image(
            cleaned_frame,
            restricted_polygon,
            detections,
            frame_violations,
            location,
            current_time,
            output_path=None,
        )
        writer.write(annotated_frame)

    capture.release()
    writer.release()

    confirmed_violations = [
        {
            "track_id": track_id,
            "dwell_seconds": round(track_dwell_seconds.get(track_id, 0), 1),
            "violation": track_last_violation.get(track_id),
        }
        for track_id in confirmed_track_ids
    ]
    track_summary = [
        {
            "track_id": track_id,
            "dwell_seconds": round(dwell, 1),
            "confirmed": track_id in confirmed_track_ids,
        }
        for track_id, dwell in track_dwell_seconds.items()
    ]

    intelligence_results = {}
    if confirmed_violations:
        intelligence_results = get_historical_context(
            location,
            current_time,
            [cv["violation"] for cv in confirmed_violations if cv["violation"]],
        )
        intelligence_results.pop("enriched_violations", None)

    return {
        "timestamp": current_time,
        "location": location,
        "sampled_fps": VIDEO_SAMPLE_FPS,
        "dwell_threshold_seconds": DWELL_CONFIRMATION_SECONDS,
        "processed_frame_count": processed_frame_count,
        "confirmed_violations": confirmed_violations,
        "track_summary": track_summary,
        "annotated_video": f"uploads/{annotated_video_filename}",
        "intelligence_data": intelligence_results,
    }