import cv2
import numpy as np
import os
from ultralytics import YOLO

MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8s.pt")
try:
    model = YOLO(MODEL_NAME)
except Exception:
    model = YOLO("yolov8s.pt")

def _hough_stripe_detect(road_crop, y_start, w, h, bright_thresh, min_clusters, min_run):
    gray = cv2.cvtColor(road_crop, cv2.COLOR_BGR2GRAY)

    _, white_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.bitwise_and(edges, white_mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=15,
        maxLineGap=15,
    )
    if lines is None:
        return None

    horiz = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 35 or angle > 145:
            horiz.append((x1, y1 + y_start, x2, y2 + y_start))

    if len(horiz) < 4:
        return None

    horiz.sort(key=lambda l: (l[1] + l[3]) // 2)
    clusters, cluster = [], [horiz[0]]
    for line in horiz[1:]:
        mid_y     = (line[1]      + line[3])      // 2
        prev_mid_y = (cluster[-1][1] + cluster[-1][3]) // 2
        if abs(mid_y - prev_mid_y) <= 20:
            cluster.append(line)
        else:
            clusters.append(cluster)
            cluster = [line]
    clusters.append(cluster)

    stripe_clusters = [c for c in clusters if len(c) >= 2]
    if len(stripe_clusters) < min_clusters:
        return None

    def cmid(c):
        return int(np.mean([(l[1] + l[3]) // 2 for l in c]))

    best_run, cur = [], [stripe_clusters[0]]
    for sc in stripe_clusters[1:]:
        if cmid(sc) - cmid(cur[-1]) <= 80:
            cur.append(sc)
        else:
            if len(cur) > len(best_run):
                best_run = cur
            cur = [sc]
    if len(cur) > len(best_run):
        best_run = cur

    if len(best_run) < min_run:
        return None

    all_lines = [l for c in best_run for l in c]
    
    pts = []
    for l in all_lines:
        pts.append([l[0], l[1]])
        pts.append([l[2], l[3]])
    pts = np.array(pts, dtype=np.int32)

    hull = cv2.convexHull(pts)
    epsilon = 0.03 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) >= 3:
        points = approx.reshape(-1, 2)
        cx, cy = np.mean(points, axis=0)
        def angle_from_center(p):
            return np.arctan2(p[1] - cy, p[0] - cx)
        sorted_points = sorted(points, key=angle_from_center)
        return np.array(sorted_points, dtype=np.int32)

    xs = pts[:, 0]
    ys = pts[:, 1]
    x_min = max(0,     int(min(xs)) - 5)
    x_max = min(w - 1, int(max(xs)) + 5)
    y_min = max(0,     int(min(ys)) - 5)
    y_max = min(h - 1, int(max(ys)) + 5)

    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.int32,
    )


def detect_restricted_zone(img):
    h, w = img.shape[:2]

    y_start = int(h * 0.30)
    road = img[y_start:, :]

    passes = [
        (200, 3, 3),
        (160, 3, 3),
        (130, 2, 2),
    ]
    for bright_thresh, min_clusters, min_run in passes:
        result = _hough_stripe_detect(road, y_start, w, h, bright_thresh, min_clusters, min_run)
        if result is not None:
            return result

    print("Zebra crossing not detected in any pass — using default zone.")

    return np.array(
        [
            [int(w * 0.15), int(h * 0.70)],
            [int(w * 0.85), int(h * 0.70)],
            [int(w * 0.95), int(h * 0.95)],
            [int(w * 0.05), int(h * 0.95)],
        ],
        np.int32,
    )


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

    cv2.rectangle(
        image,
        (x, box_top),
        (box_right, box_bottom),
        color,
        -1,
    )
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


def process_frame(frame_path, conf_threshold=0.25):
    img = cv2.imread(frame_path)

    if img is None:
        return {"error": "Image not found"}

    restricted_polygon = detect_restricted_zone(img)
    
    allowed_classes = [1, 2, 3, 5, 7]
    
    results = model(img, conf=conf_threshold, classes=allowed_classes)[0]

    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append(
            {
                "vehicle_type": model.names[cls_id],
                "confidence": round(float(box.conf[0]) * 100, 2),
                "bbox": [x1, y1, x2, y2],
            }
        )

    output_path = "output_evidence.jpg"

    return {
        "status": "success",
        "detections": detections,
        "restricted_polygon": restricted_polygon.tolist(),
        "image_shape": img.shape,
        "evidence_image": output_path,
    }


def generate_evidence_image(
    frame_path,
    restricted_polygon,
    detections,
    violations,
    location_name,
    current_time,
    intelligence_data=None,
    output_path="output_evidence.jpg",
):
    img = cv2.imread(frame_path)

    if img is None:
        return None

    polygon = np.array(restricted_polygon, dtype=np.int32)

    violation_lookup = {tuple(v["bbox"]): v for v in violations}

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        violation = violation_lookup.get(tuple(det["bbox"]))

        if violation:
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 220), 3)
            
            box_h = y2 - y1
            y1_contact = int(y2 - 0.20 * box_h)
            
            cv2.rectangle(img, (x1, y1_contact), (x2, y2), (0, 140, 255), 1)
            
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1_contact), (x2, y2), (0, 140, 255), -1)
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

            label = f"{violation['violation_type']} | {det['vehicle_type']} {violation['violation_confidence']:.1f}%"
            _draw_label(img, label, (x1, max(22, y1)), (40, 40, 220))
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 180, 75), 2)
            label = f"{det['vehicle_type']} {det['confidence']:.1f}%"
            _draw_label(img, label, (x1, max(22, y1)), (60, 180, 75))

    cv2.polylines(img, [polygon], True, (255, 0, 0), 3)

    summary_lines = [
        f"Location: {location_name}",
        f"Time: {current_time}",
        f"Violations: {len(violations)}",
    ]

    if intelligence_data:
        risk_score = intelligence_data.get("risk_score")
        recommendation = intelligence_data.get("recommendation")
        hotspot_flag = intelligence_data.get("predicted_congestion_hotspot")

        if risk_score is not None:
            summary_lines.append(f"Risk Score: {risk_score}")
        if hotspot_flag is not None:
            summary_lines.append(
                "Congestion Risk: HIGH" if hotspot_flag else "Congestion Risk: LOW"
            )
        if recommendation:
            summary_lines.append(f"Action: {recommendation}")

    overlay_height = 34 + (len(summary_lines) * 24)
    cv2.rectangle(img, (10, 10), (min(img.shape[1] - 10, 700), overlay_height), (24, 24, 24), -1)

    for index, line in enumerate(summary_lines):
        cv2.putText(
            img,
            line,
            (22, 34 + (index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

    if not violations:
        cv2.putText(
            img,
            "No parking violation detected in restricted zone",
            (22, min(img.shape[0] - 30, overlay_height + 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (30, 200, 30),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_path, img)
    return output_path