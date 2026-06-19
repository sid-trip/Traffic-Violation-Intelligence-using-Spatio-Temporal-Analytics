import pandas as pd
from datetime import datetime

try:
    df = pd.read_csv("historical_violations.csv")
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df['hour'] = df['created_datetime'].dt.hour
except Exception as e:
    print(f"Warning: Could not load dataset. Mocking data for now. {e}")
    df = None

def get_historical_context(location_name: str, current_time: str):
    """
    Looks up the location in the 298k dataset, checks the specific hour, 
    and returns a risk score and recommendation.
    """
    if df is None:
        # Fallback for testing without data
        return {"risk_score": 85, "historical_count": 42, "recommendation": "Dispatch Tow Truck"}

    # Parse current hour
    current_hour = datetime.strptime(current_time, "%H:%M").hour

    # Filter data for this location and this specific hour
    local_data = df[(df['location'].str.contains(location_name, case=False, na=False)) & 
                    (df['hour'] == current_hour)]
    
    historical_count = len(local_data)

    # Simple logic to determine risk
    if historical_count > 50:
        risk_score = 90
        rec = "CRITICAL: High probability of gridlock. Dispatch Tow Truck immediately."
    elif historical_count > 20:
        risk_score = 65
        rec = "WARNING: Moderate spillover risk. Alert nearest patrol."
    else:
        risk_score = 30
        rec = "NORMAL: Monitor via CCTV. Auto-generate e-challan."

    return {
        "risk_score": risk_score,
        "historical_count": historical_count,
        "recommendation": rec
    }