# Contextual Traffic Enforcement and Parking Intelligence Platform

This repository contains the minimum viable product (MVP) implementation for a context-aware traffic vision platform. The system is designed to analyze traffic-camera imagery, detect vehicles, evaluate spatial violations against predefined road boundaries (such as zebra crossings and footpaths), and enrich the annotated output with historical congestion risk assessments.

---

## Technical Overview

The system transitions traffic enforcement from simple object detection to multi-stage spatial and historical decision support. 

### Data Flow Pipeline
1. **Frame Ingestion:** The edge node processes a target image feed (`test_image.jpg` or manual file upload) along with metadata (timestamp, target camera node).
2. **Object Detection:** A YOLOv8 model localizes road users (vehicles, motorcycles, buses, trucks, and auto-rickshaws).
3. **Spatial Intersection:** An OpenCV-based spatial rule engine maps the detections against camera-specific semantic regions (such as a zebra crossing) using `cv2.pointPolygonTest` to evaluate coordinate boundaries.
4. **Contextual Enrichment:** The system queries `intelligence.py` to match the current location and hour against structured records from the Bengaluru Traffic Police (BTP) historical datasets on OpenCity.
5. **Evidence Packaging:** The renderer produces a structured metadata payload (JSON format) and outputs an annotated evidence image (`output_evidence.jpg`) with bounding boxes, spatial region outlines, and an operational action summary.

---

## Directory Structure

```text
.
├── app.py
├── intelligence.py
├── historical_violations.csv
├── event_data_traffic_violations.csv
├── test_image.jpg
├── requirements.txt
└── README.md
```

---

## Requirements

The execution suite requires the following software environment:
* Python 3.8 or higher
* Pip package manager

---

## Installation

Run the following commands in your terminal to configure the execution environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Requirements Configuration

Your `requirements.txt` file must contain the following package declarations:

```text
streamlit
ultralytics
opencv-python-headless
numpy
pandas
pillow
```

---

## Input File Configuration

To execute inference, ensure the following datasets and files are placed in the root directory:

### 1. Default Telemetry Frame
Place an image named `test_image.jpg` in the root directory. This image serves as the default edge input if no live image is manually uploaded.

### 2. Historical Violation Database (`historical_violations.csv`)
This structured file must contain historical data parameters published by the Bengaluru Traffic Police on OpenCity with the following column headers:
* `location`
* `junction_name`
* `police_station`
* `vehicle_type`
* `violation_type`
* `created_datetime`

### 3. Traffic Event Logs (`event_data_traffic_violations.csv`)
This structured file must contain recent congestion and incident logs with the following column headers:
* `address`
* `junction`
* `police_station`
* `event_cause`
* `priority`
* `start_datetime`

---

## How to Run

Follow these instructions to start the processing node and open the contextual intelligence dashboard.

### 1. Activating the Virtual Environment
```bash
source venv/bin/activate
```

### 2. Running the Streamlit Interface
Execute the Streamlit application to spin up the local webserver:
```bash
streamlit run app.py
```

### 3. Accessing the Dashboard
Once the server starts, open your browser and navigate to the local network port printed in your terminal (typically `http://localhost:8501`).

---

## System Usage Guide

### Node Selection
Use the sidebar drop-down menu to configure the target camera node (e.g., `CAM_001_INDIRANAGAR`). The system uses this location metadata to filter the historical BTP database.

### Detection Threshold
Adjust the slider to filter YOLOv8 object detections based on confidence score. Detections below this threshold will be pruned before spatial logic is applied.

### Data Upload
The system defaults to `test_image.jpg` if present in the workspace. To evaluate a new frame, use the drag-and-drop file uploader to ingest an alternative JPEG or PNG source frame.

### Executing Inference
Click the primary action button labeled **Execute Spatial Inference**. The execution cycle will run, producing:
* **Left Panel:** The raw edge node source imagery.
* **Right Panel:** The compiled evidence image displaying red bounding boxes for violators, green bounding boxes for compliant road users, and blue polygons for restricted boundaries.
* **Execution Logs:** Verbose system-level logs showing the internal state of the rule engine and spatial calculations.
* **JSON Metadata Payload:** The raw data structure returned by the analytics engine, ready for API transmission to central municipal command hubs.