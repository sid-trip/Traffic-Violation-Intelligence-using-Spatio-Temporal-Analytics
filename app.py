import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

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

    /* Page title block, set in mono like an instrument header */
    .console-title {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 1.6rem;
        letter-spacing: 0.01em;
        color: var(--text);
        margin-bottom: 0.1rem;
    }
    .console-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: var(--muted);
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    hr, .console-rule {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.4rem 0;
    }

    /* Step rail headers */
    .step-header {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-top: 0.6rem;
        margin-bottom: 0.1rem;
    }
    .step-number {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 0.78rem;
        color: var(--bg);
        background: var(--ok);
        border-radius: 3px;
        padding: 0.12rem 0.5rem;
        letter-spacing: 0.03em;
    }
    .step-title {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 1.05rem;
        color: var(--text);
    }
    .step-caption {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
        color: var(--muted);
        margin-bottom: 0.8rem;
    }

    /* Monitor-style frame around images */
    .monitor-frame {
        border: 1px solid var(--border);
        background: var(--panel);
        border-radius: 4px;
        padding: 8px 8px 0 8px;
        margin-bottom: 0.3rem;
    }
    .monitor-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0 2px 6px 2px;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .rec-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--alert);
        display: inline-block;
    }

    /* Status chips replacing default st.success/warning boxes */
    .status-chip {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        border-radius: 4px;
        padding: 0.55rem 0.9rem;
        border: 1px solid var(--border);
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .status-chip.ok { color: var(--ok); border-color: rgba(45,212,167,0.35); background: rgba(45,212,167,0.07); }
    .status-chip.warn { color: var(--warn); border-color: rgba(255,180,84,0.35); background: rgba(255,180,84,0.07); }

    /* Metric readout strip */
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--muted);
    }

    /* Violation cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border-color: var(--border) !important;
        border-radius: 4px;
    }

    /* Buttons */
    .stButton > button {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.03em;
        border-radius: 4px;
        border: 1px solid var(--border);
    }
    .stButton > button[kind="primary"] {
        background: var(--ok);
        color: #06140F;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background: #25C29A;
    }

    /* Expanders */
    details {
        background: var(--panel);
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
    }

    /* Code blocks (logs) */
    .stCodeBlock, pre {
        font-family: 'IBM Plex Mono', monospace !important;
        border: 1px solid var(--border) !important;
    }

    /* Captions throughout */
    [data-testid="stCaptionContainer"], .stCaption {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--muted) !important;
    }
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

def _hough_stripe_detect(road_crop, y_start, w, h, bright_thresh, min_clusters, min_run):
    gray = cv2.cvtColor(road_crop, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    white_mask = cv2.dilate(white_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
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

    xmids = [(l[0] + l[2]) // 2 for l in horiz]
    median_x = np.median(xmids)
    horiz = [l for l in horiz if abs((l[0] + l[2]) // 2 - median_x) < (w * 0.35)]

    if len(horiz) < 4:
        return None

    horiz.sort(key=lambda l: (l[1] + l[3]) // 2)
    clusters, cluster = [], [horiz[0]]
    for line in horiz[1:]:
        mid_y = (line[1] + line[3]) // 2
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
    xs = pts[:, 0]
    ys = pts[:, 1]
    bbox_w = int(max(xs)) - int(min(xs))
    bbox_h = int(max(ys)) - int(min(ys))

    if bbox_h == 0 or (bbox_w / bbox_h) < 1.8:
        return None  # too square/tall to be a stripe band
    if bbox_w < w * 0.35:
        return None  # too narrow to plausibly span the carriageway
    band_center_y = (int(min(ys)) + int(max(ys))) / 2
    if band_center_y < road_crop.shape[0] * 0.40:
        return None  # sits too high in the crop to be near-camera road surface
    cluster_mids = sorted(cmid(c) for c in best_run)
    gaps = np.diff(cluster_mids)
    if len(gaps) >= 2:
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)
        if gap_mean > 0 and (gap_std / gap_mean) > 0.55:
            return None  # stripes should repeat at a roughly regular interval
    # --- end geometric checks --------------------------------------------

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
    x_min = max(0, int(min(xs)) - 5)
    x_max = min(w - 1, int(max(xs)) + 5)
    y_min = max(0, int(min(ys)) - 5)
    y_max = min(h - 1, int(max(ys)) + 5)
    return np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.int32,
    )


def detect_restricted_zone(img):
    """Returns (polygon, was_auto_detected). was_auto_detected is False when
    the function had to fall back to the generic guessed trapezoid.

    The fallback box is intentionally narrower and lower than a naive guess
    might be: it sits close to the camera, near where vehicles are most
    likely to be standing in a typical traffic-camera frame, which gives
    the operator a more useful starting point to drag/redraw from."""
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
        image,
        text,
        (x + padding, box_bottom - baseline - padding),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def run_detection(img, conf_threshold=0.25):
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
    return detections


def save_evidence_image(annotated_img, output_path="output_evidence.png"):
    """Write the annotated image losslessly.

    PNG avoids the extra quality loss that comes from re-encoding the image as
    JPEG, which is the main reason output evidence often looks soft or degraded.
    """
    if output_path.lower().endswith(".png"):
        cv2.imwrite(output_path, annotated_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(output_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return output_path


def generate_evidence_image(
    img,
    restricted_polygon,
    detections,
    violations,
    location_name,
    current_time,
    intelligence_data=None,
    output_path="output_evidence.png",
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

    # Removed the top-left metadata overlay on request.

    if not violations:
        no_violation_text = "No parking violation detected in restricted zone"
        x, y = 22, min(annotated.shape[0] - 20, 40)
        cv2.putText(
            annotated,
            no_violation_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (30, 200, 30),
            2,
            cv2.LINE_AA,
        )

    return save_evidence_image(annotated, output_path)

def render_zone_preview(img, polygon):
    preview = img.copy()
    cv2.polylines(preview, [np.array(polygon, dtype=np.int32)], True, (255, 0, 0), 3)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def render_click_progress(cv_img, points):
    """Draw the points placed so far, numbered, connected by lines, with a
    thin closing-preview line back to the start once 3+ points exist --
    so the operator sees the zone taking shape while clicking instead of
    clicking blind on an unmarked image."""
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


# UI
st.markdown('<div class="console-title">Contextual Traffic Enforcement System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="console-subtitle">Spatial Overlap Inference &amp; Violation Classification · MVP</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="console-rule">', unsafe_allow_html=True)

# --- Sidebar: configuration only. No zone/result content lives here. ---
with st.sidebar:
    st.header("Node Configuration")
    camera_key = st.selectbox("Select Target Node", list(CAMERA_NODES.keys()))
    location_name = CAMERA_NODES[camera_key]
    confidence_thresh = st.slider("Detection Confidence Threshold", 0.0, 1.0, 0.45)

    st.caption(
        "Lower this if vehicles aren't being picked up; raise it to reduce false "
        "detections. The violation rule engine applies its own additional "
        "minimum-confidence floor on top of this."
    )

    st.markdown("---")
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
    selected_image = None
    cv_img_raw = None

    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        selected_image = Image.open(uploaded_file).convert("RGB")
        cv_img_raw = cv2.imdecode(np.frombuffer(upload_bytes, np.uint8), cv2.IMREAD_COLOR)
        if cv_img_raw is None:
            st.error("Could not decode the uploaded image.")
    else:
        if os.path.exists("test_image.jpg"):
            cv_img_raw = cv2.imread("test_image.jpg")
            if cv_img_raw is not None:
                selected_image = Image.open("test_image.jpg").convert("RGB")
                st.write("Status: Operating on default telemetry frame (test_image.jpg).")
            else:
                st.error("test_image.jpg could not be read.")
        else:
            st.write("Status: Direct image upload required. (test_image.jpg not located).")

# Reset cached zone + result state if the input image changes
if selected_image is not None:
    image_fingerprint = (uploaded_file.name if uploaded_file else "test_image.jpg", selected_image.size)
    if st.session_state.get("_image_fingerprint") != image_fingerprint:
        st.session_state["_image_fingerprint"] = image_fingerprint
        st.session_state.pop("restricted_polygon", None)
        st.session_state.pop("zone_auto_detected", None)
        st.session_state.pop("manual_points", None)
        st.session_state.pop("last_result", None)

if selected_image is None or cv_img_raw is None:
    st.info("Awaiting input. Select a target node and provide a source image in the sidebar to begin.")
    st.stop()

# Run auto-detection once per image and cache it
if "restricted_polygon" not in st.session_state:
    auto_poly, was_auto = detect_restricted_zone(cv_img_raw)
    st.session_state["restricted_polygon"] = auto_poly
    st.session_state["zone_auto_detected"] = was_auto

# ===========================================================================
# STEP 1 -- Calibrate Zone
# ===========================================================================
st.markdown(
    '<div class="step-header"><span class="step-number">STEP 1</span>'
    '<span class="step-title">Calibrate Restricted Zone</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="step-caption">Target node: <b>{camera_key}</b> &nbsp;·&nbsp; '
    f'Resolved location: <b>{location_name}</b></div>',
    unsafe_allow_html=True,
)

cal_col1, cal_col2 = st.columns(2)
with cal_col1:
    st.markdown(
        '<div class="monitor-frame"><div class="monitor-label">'
        '<span class="rec-dot"></span>SOURCE IMAGE · EDGE NODE</div>',
        unsafe_allow_html=True,
    )
    st.image(selected_image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with cal_col2:
    st.markdown(
        '<div class="monitor-frame"><div class="monitor-label">ZONE PREVIEW</div>',
        unsafe_allow_html=True,
    )
    st.image(
        render_zone_preview(cv_img_raw, st.session_state["restricted_polygon"]),
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state["zone_auto_detected"]:
    st.markdown(
        '<div class="status-chip ok">✓ &nbsp;ZONE AUTO-DETECTED FROM STRIPE PATTERN</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-chip warn">⚠ &nbsp;STRIPE PATTERN NOT CONFIDENTLY DETECTED — '
        'SHOWING ESTIMATED REGION. VERIFY OR REDRAW BELOW.</div>',
        unsafe_allow_html=True,
    )

with st.expander("Correct / calibrate this zone manually", expanded=not st.session_state["zone_auto_detected"]):
    st.caption(
        "In a real deployment this calibration is done once per camera "
        "and saved (see PS3 doc, section 6.1 'Field Layer'). Here you "
        "can correct it per-image for the demo."
    )
    if HAS_CLICK_WIDGET:
        st.write("Click 4 corners on the image below, in order, to define the zone.")
        if "manual_points" not in st.session_state:
            st.session_state["manual_points"] = []

        # Render the image WITH the points/lines placed so far, so the
        # operator gets live feedback instead of clicking on a blank image
        # and only seeing the result after all 4 points are placed.
        preview_rgb = render_click_progress(cv_img_raw, st.session_state["manual_points"])
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
                st.session_state["restricted_polygon"] = np.array(
                    st.session_state["manual_points"], dtype=np.int32
                )
                st.session_state["zone_auto_detected"] = True  # operator-confirmed, treat as trustworthy
                st.session_state["manual_points"] = []
                st.rerun()
    else:
        st.info(
            "Install `streamlit-image-coordinates` for click-to-draw "
            "zone calibration (`pip install streamlit-image-coordinates`). "
            "Falling back to numeric corner entry."
        )
        h, w = cv_img_raw.shape[:2]
        default_poly = st.session_state["restricted_polygon"]
        pts = []
        for i in range(4):
            dx = int(default_poly[i][0]) if i < len(default_poly) else int(w * 0.5)
            dy = int(default_poly[i][1]) if i < len(default_poly) else int(h * 0.5)
            c1, c2 = st.columns(2)
            with c1:
                px = st.number_input(f"Point {i+1} X", 0, w, dx, key=f"px_{i}")
            with c2:
                py = st.number_input(f"Point {i+1} Y", 0, h, dy, key=f"py_{i}")
            pts.append((px, py))
        if st.button("Apply manual zone", type="primary"):
            st.session_state["restricted_polygon"] = np.array(pts, dtype=np.int32)
            st.session_state["zone_auto_detected"] = True
            st.rerun()

st.markdown('<hr class="console-rule">', unsafe_allow_html=True)

# ===========================================================================
# STEP 2 -- Run Inference
# ===========================================================================
st.markdown(
    '<div class="step-header"><span class="step-number">STEP 2</span>'
    '<span class="step-title">Run Inference</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="step-caption">Runs detection, spatial rule evaluation, and '
    'historical risk lookup using the zone above.</div>',
    unsafe_allow_html=True,
)

if st.button("Execute Spatial Inference", type="primary", use_container_width=True):
    with st.spinner("Running detection, rule evaluation, and historical lookups..."):
        cv_img = cv_img_raw.copy()
        restricted_polygon = st.session_state["restricted_polygon"]

        detections = run_detection(cv_img, confidence_thresh)
        current_time_str = datetime.now().strftime("%H:%M")

        rule_engine = ParkingRuleEngine(restricted_polygon, location_name, current_time_str)
        violations = rule_engine.evaluate(detections, cv_img.shape)

        intelligence_data = get_historical_context(location_name, current_time_str, violations)
        enriched_violations = intelligence_data["enriched_violations"]

        output_image_path = generate_evidence_image(
            cv_img,
            restricted_polygon,
            detections,
            enriched_violations,
            location_name,
            current_time_str,
            intelligence_data,
        )

        # Cache everything needed to render the result section, so it
        # survives reruns triggered by widgets further down the page.
        st.session_state["last_result"] = {
            "output_image_path": output_image_path,
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
# RESULT -- only appears once Step 2 has been run at least once
# ===========================================================================
if "last_result" not in st.session_state:
    st.markdown(
        '<div class="step-header"><span class="step-title">Result</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="step-caption">Run inference above to see the annotated '
        'evidence image and risk assessment here.</div>',
        unsafe_allow_html=True,
    )
else:
    r = st.session_state["last_result"]
    zone_ctx = r["zone_context"]
    intelligence_data = r["intelligence_data"]
    enriched_violations = r["enriched_violations"]

    risk_score = intelligence_data["risk_score"]
    risk_color = "var(--alert)" if risk_score >= 70 else ("var(--warn)" if risk_score >= 45 else "var(--ok)")

    st.markdown(
        '<div class="step-header"><span class="step-title">Result</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="step-caption" style="font-family:\'IBM Plex Mono\',monospace;'
        f'font-size:0.8rem;">'
        f'<span class="rec-dot" style="background:{risk_color};"></span>&nbsp; '
        f'RISK {risk_score}% &nbsp;·&nbsp; {len(enriched_violations)} VIOLATION(S) &nbsp;·&nbsp; '
        f'{r["location_name"].upper()} &nbsp;·&nbsp; {datetime.now().strftime("%H:%M")}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="monitor-frame"><div class="monitor-label">'
        '<span class="rec-dot"></span>ANNOTATED OUTPUT · INFERENCE RESULT</div>',
        unsafe_allow_html=True,
    )
    # Keep the output at its native resolution as much as Streamlit/browser layout allows.
    # Do not force container-width scaling here.
    st.image(r["output_image_path"], use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        f"Zone rule: **{zone_ctx['violation_type']}** "
        f"({zone_ctx['rule_source']} → {zone_ctx['matched_on']}) | "
        f"Active window: {zone_ctx['rule_window']} | "
        f"Currently active: {'Yes' if zone_ctx['rule_active'] else 'No (low-confidence)'}"
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Violations Detected", len(enriched_violations))
    metric_col2.metric("Risk Score", f"{intelligence_data['risk_score']}%")
    metric_col3.metric(
        "Congestion Risk",
        "HIGH" if intelligence_data.get("predicted_congestion_hotspot") else "LOW",
    )

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
