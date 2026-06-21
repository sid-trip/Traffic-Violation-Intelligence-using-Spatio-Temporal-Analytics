from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import shutil
import os

from intelligence import get_historical_context
from rules import ParkingRuleEngine

from fastapi.staticfiles import StaticFiles
import uuid

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

@app.post("/analyze")
async def analyze_traffic(
    image: UploadFile = File(...),
    location: str = Form(...),
    timestamp: str = Form(None)
):
    # Generate a unique prefix to prevent collisions and caching issues
    unique_id = uuid.uuid4().hex[:8]
    sanitized_filename = f"{unique_id}_{image.filename}"
    file_path = os.path.join(UPLOAD_DIR, sanitized_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    if timestamp:
        current_time = datetime.fromisoformat(timestamp).strftime("%H:%M")
    else:
        current_time = datetime.now().strftime("%H:%M")

    vision_results = process_frame(file_path)

    if "error" in vision_results:
        return vision_results

    rule_engine = ParkingRuleEngine(
        vision_results["restricted_polygon"],
        location,
        current_time
    )

    violations = rule_engine.evaluate(
        vision_results["detections"],
        vision_results["image_shape"]
    )

    intelligence_results = {}

    if len(violations) > 0:
        intelligence_results = get_historical_context(
            location,
            current_time,
            violations
        )
        violations = intelligence_results.get(
            "enriched_violations",
            violations
        )
        intelligence_results.pop(
            "enriched_violations",
            None
        )

    # Generate a unique path for the annotated evidence image
    base_name, ext = os.path.splitext(sanitized_filename)
    evidence_filename = f"{base_name}_evidence{ext}"
    evidence_file_path = os.path.join(UPLOAD_DIR, evidence_filename)

    evidence_image = generate_evidence_image(
        file_path,
        vision_results["restricted_polygon"],
        vision_results["detections"],
        violations,
        location,
        current_time,
        intelligence_results,
        output_path=evidence_file_path
    )

    if evidence_image:
        vision_results["evidence_image"] = f"uploads/{evidence_filename}"

    return {
        "timestamp": current_time,
        "location": location,
        "violations": violations,
        "vision_data": vision_results,
        "intelligence_data": intelligence_results
    }
