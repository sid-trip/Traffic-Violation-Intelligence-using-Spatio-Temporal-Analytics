import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

def auto_detect_zebra_crossing(img):
    """
    Analyzes the image using classical Computer Vision to locate the 
    zebra crossing stripes dynamically and return a polygon.
    """
    height, width = img.shape[:2]
    
    # 1. Focus on the lower half of the image (where the road usually is)
    roi_y_start = int(height * 0.5)
    roi_img = img[roi_y_start:height, 0:width]
    
    # 2. Convert to grayscale and blur
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Threshold to find bright white stripes
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    
    # 4. Find contours of the white stripes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stripe_points = []
    
    for cnt in contours:
        # Filter out noise based on size and aspect ratio
        area = cv2.contourArea(cnt)
        if 200 < area < 15000:  # Tune these limits if needed
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            
            # Zebra crossing stripes are usually wider than they are tall
            if aspect_ratio > 1.2:
                # Add these contour points (shifting y back to original image coordinates)
                for pt in cnt:
                    stripe_points.append([pt[0][0], pt[0][1] + roi_y_start])
                    
    # 5. If we found enough stripe points, find their Convex Hull (boundary polygon)
    if len(stripe_points) > 10:
        pts = np.array(stripe_points, np.int32)
        hull = cv2.convexHull(pts)
        
        # Simplify the hull to a 4-point polygon (quadrilateral)
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) >= 4:
            # Sort points to make a clean quad polygon
            approx = approx.reshape(-1, 2)
            return approx
            
    # Fallback to a default zone if no zebra crossing is detected
    return np.array([
        [int(width*0.10), int(height*0.75)],
        [int(width*0.90), int(height*0.75)],
        [int(width*0.85), int(height*0.95)],
        [int(width*0.15), int(height*0.95)]
    ], np.int32)


def process_frame(frame_path: str, location_name: str):
    img = cv2.imread(frame_path)
    if img is None:
        return {"error": "Image not found"}

    # THE MAGIC LAYER: Dynamically detect the zebra crossing
    restricted_polygon = auto_detect_zebra_crossing(img)

    results = model(img)[0]
    violations_detected = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in [2, 3, 5, 7]: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Tire Footprint logic
            cx = (x1 + x2) // 2
            footprint_y = y2 - 5 
            
            # Check if inside our dynamically detected polygon
            is_inside = cv2.pointPolygonTest(restricted_polygon, (cx, footprint_y), False)
            
            if is_inside >= 0:
                violations_detected.append({
                    "vehicle_type": model.names[cls_id],
                    "confidence": round(confidence * 100, 2),
                    "violation_type": "Stop-Line / Zebra Crossing Violation",
                    "bbox": [x1, y1, x2, y2]
                })
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, "VIOLATION", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw footprint debug dot
            cv2.circle(img, (cx, footprint_y), 5, (0, 255, 255), -1)

    # Draw the Dynamically Generated Polygon in Blue
    cv2.polylines(img, [restricted_polygon], isClosed=True, color=(255, 0, 0), thickness=3)
    
    # Add transparent red overlay to show the "active" zone
    overlay = img.copy()
    cv2.fillPoly(overlay, [restricted_polygon], (0, 0, 255))
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    output_path = "output_evidence.jpg"
    cv2.imwrite(output_path, img)

    return {
        "status": "success",
        "violations": violations_detected,
        "evidence_image": output_path
    }