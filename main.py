from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from intelligence import get_historical_context
from vision import process_frame

app = FastAPI()

# Allow frontend HTML/JS to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze")
def analyze_traffic(image_path: str = "test_img.jpg", location: str = "Indiranagar"):
    """
    The main API endpoint. 
    1. Runs vision processing.
    2. Looks up historical dataset.
    3. Returns full JSON package.
    """
    current_time = datetime.now().strftime("%H:%M")
    
    # 1. Run Vision Layer
    vision_results = process_frame(image_path, location)
    
    if "error" in vision_results:
        return {"error": vision_results["error"]}

    # 2. Run Intelligence Layer
    # If violations were found, query the 298k dataset for risk assessment
    intelligence_results = {}
    if len(vision_results["violations"]) > 0:
        intelligence_results = get_historical_context(location, current_time)

    # 3. Compile Final Response
    return {
        "timestamp": current_time,
        "location": location,
        "vision_data": vision_results,
        "intelligence_data": intelligence_results
    }

# To run the server, type this in terminal:
# uvicorn main:app --reload