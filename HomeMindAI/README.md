# HomeMindAI

HomeMindAI is a production-ready foundation for a futuristic AI-powered home intelligence surveillance platform. Part 3 extends the platform with YOLOv8 person detection, InsightFace recognition, known-vs-unknown identity matching, lightweight tracking, structured AI events, and unknown visitor capture.

## Project Overview

This foundation is intentionally modular so each subsystem can evolve independently:

- `app/camera/` contains camera capture services and future stream adapters.
- `app/vision/` contains the frame pipeline, motion detection, YOLO detector, and tracking foundation.
- `app/identity/` contains face recognition, embeddings, matching, and registry loading.
- `app/events/` contains structured perception event generation.
- `app/alerts/` will handle notifications and automation triggers.
- `app/memory/` will support long-term AI context and retrieval.
- `app/agents/` is reserved for autonomous orchestration logic.
- `app/dashboard/` is the future API and UI integration layer.
- `app/utils/` contains shared utilities such as logging and image helpers.
- `data/` stores faces, snapshots, and log artifacts.
- `models/` stores local AI weights and model metadata.
- `config/` centralizes settings and environment loading.

## Part 3 Features

- Webcam camera ingestion for local testing.
- RTSP stream support for CCTV-style camera feeds.
- Multi-camera-ready manager and stream abstraction.
- Frame resizing and FPS limiting.
- Graceful reconnect handling for unstable sources.
- Lightweight motion detection using frame differencing and contour analysis.
- YOLOv8 person detection with confidence filtering.
- InsightFace face embeddings and cosine-similarity identity matching.
- Known vs unknown recognition using the project-level `Photos/` dataset.
- Lightweight tracking IDs ready for a future DeepSORT upgrade.
- Structured perception events for known, unknown, and tracked people.
- Automatic unknown-face capture stored under `data/unknown_faces/`.
- Automatic motion snapshots stored under `data/snapshots/`.
- Manual snapshot capture through an API endpoint.
- MJPEG live streaming through FastAPI.
- CPU-friendly inference stride support with future GPU device configuration.

## Folder Structure

```text
HomeMindAI/
├── app/
│   ├── agents/
│   ├── alerts/
│   ├── camera/
│   ├── dashboard/
│   ├── events/
│   ├── identity/
│   ├── memory/
│   ├── utils/
│   └── vision/
├── config/
├── data/
│   ├── known_faces/
│   ├── logs/
│   ├── snapshots/
│   └── unknown_faces/
├── models/
├── tests/
├── .env.example
├── main.py
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.10
- A working webcam
- Optional RTSP camera URL for CCTV testing
- Internet access on the first run if YOLO or InsightFace models need to auto-download
- macOS, Linux, or Windows camera permissions enabled for Python/terminal access

## Setup Instructions

### 1. Enter the project

```bash
cd HomeMindAI
```

### 2. Create a virtual environment

```bash
python3.10 -m venv .venv
```

### 3. Activate the virtual environment

On macOS and Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Create your environment file

```bash
cp .env.example .env
```

### 6. Configure the camera source

For webcam mode:

```env
HOMEMINDAI_CAMERA_SOURCE=webcam
HOMEMINDAI_CAMERA_DEVICE_INDEX=0
```

For RTSP mode:

```env
HOMEMINDAI_CAMERA_SOURCE=rtsp
HOMEMINDAI_RTSP_URL=rtsp://username:password@ip:554/stream
```

Common video settings:

```env
HOMEMINDAI_FRAME_WIDTH=960
HOMEMINDAI_FRAME_HEIGHT=540
HOMEMINDAI_FPS_LIMIT=12
HOMEMINDAI_RECONNECT_INTERVAL_SECONDS=3
HOMEMINDAI_MOTION_AREA_THRESHOLD=1200
HOMEMINDAI_MOTION_DELTA_THRESHOLD=25
HOMEMINDAI_MOTION_BLUR_SIZE=21
```

AI perception settings:

```env
HOMEMINDAI_YOLO_MODEL_PATH=yolov8n.pt
HOMEMINDAI_YOLO_CONFIDENCE_THRESHOLD=0.45
HOMEMINDAI_YOLO_DEVICE=cpu
HOMEMINDAI_INSIGHTFACE_MODEL_NAME=buffalo_l
HOMEMINDAI_INSIGHTFACE_PROVIDERS=CPUExecutionProvider
HOMEMINDAI_FACE_DETECTION_WIDTH=640
HOMEMINDAI_FACE_DETECTION_HEIGHT=640
HOMEMINDAI_FACE_SIMILARITY_THRESHOLD=0.35
HOMEMINDAI_TRACKING_TIMEOUT_SECONDS=2.5
HOMEMINDAI_AI_INFERENCE_STRIDE=2
HOMEMINDAI_PHOTOS_DATASET_PATH=./Photos
HOMEMINDAI_EMBEDDING_CACHE_PATH=./data/cache/family_embeddings.pkl
HOMEMINDAI_KNOWN_FACES_PATH=./Photos
HOMEMINDAI_UNKNOWN_FACES_PATH=./data/unknown_faces
```

## Model Download Notes

- `ultralytics` will automatically download `yolov8n.pt` on first use if it is not already present.
- `InsightFace` will automatically download the configured model pack, such as `buffalo_l`, on first use.
- To use local assets instead, point `HOMEMINDAI_YOLO_MODEL_PATH` to your downloaded YOLO weights.

## Run The Camera Test

Start the local OpenCV camera test window:

```bash
python main.py
```

Behavior:

- Opens the default webcam.
- Displays the video stream in a window titled `HomeMindAI Camera Test`.
- Exits when you press `Q`.
- Writes logs to `data/logs/homemindai.log`.

If the camera does not open, confirm that your operating system has granted camera permissions and that no other app is already using the webcam.

## Run The FastAPI Service

Start the API server:

```bash
uvicorn main:app --reload
```

Available endpoints:

- `GET /health` returns service status and active camera IDs.
- `GET /camera/live` returns an MJPEG stream for the primary camera.
- `POST /camera/snapshot` stores a manual snapshot from the latest processed frame.
- `GET /events/recent` returns recent AI detection and identity events.
- `POST /identity/reload-known-faces` reloads known-face embeddings after adding or changing family images.

Open these URLs in a browser or API client:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/camera/live`

Manual snapshot example:

```bash
curl -X POST http://127.0.0.1:8000/camera/snapshot
```

Recent events example:

```bash
curl http://127.0.0.1:8000/events/recent
```

If you later add multiple cameras, the endpoints accept a `camera_id` query parameter.

## Motion Detection Overview

The motion engine uses a lightweight and fast classical computer-vision approach:

1. Convert frames to grayscale.
2. Apply Gaussian blur to reduce noise.
3. Compute frame differencing against the previous frame.
4. Threshold and dilate the motion mask.
5. Extract contours and filter them by area.
6. Save a snapshot automatically when significant motion is detected.

This keeps the system efficient while leaving room for future YOLO and face-recognition inference.

## Recognition Workflow

Each AI inference cycle follows this sequence:

1. Resize the frame.
2. Run motion analysis.
3. Run YOLOv8 person detection.
4. Assign lightweight tracking IDs.
5. Detect faces and generate InsightFace embeddings.
6. Compare live embeddings against cached family embeddings and averaged identity profiles using cosine similarity.
7. Label the person as known, unknown, or face-not-visible.
8. Emit structured events and save unknown-face snapshots when needed.

## Family Dataset Setup

HomeMindAI now scans the project-level `Photos/` directory automatically. Create family/member folders like this:

```text
Photos/
├── rishi/
│   ├── 1.jpg
│   └── 2.jpg
├── sravan/
│   ├── 1.jpg
│   └── 2.jpg
├── himaja/
│   ├── 1.jpg
│   └── 2.jpg
└── kumar/
	├── 1.jpg
	└── 2.jpg
```

Default identity mapping:

- `rishi` -> `Rishi`
- `sravan` -> `Brother`
- `himaja` -> `Mother`
- `kumar` -> `Father`

Guidelines:

- Use clear, front-facing face images when possible.
- Add multiple images per person for better matching stability.
- The loader scans person folders recursively, so nested subfolders also work.
- Invalid images are skipped automatically.
- Face embeddings are cached to speed up future startups.
- After adding new images while the server is running, call `POST /identity/reload-known-faces`.

Recommended photo quality:

- Use one face per image when possible.
- Prefer bright lighting and minimal blur.
- Include slight angle variation, but keep the face clearly visible.
- Avoid sunglasses, heavy filters, and distant group shots.

Reload example:

```bash
curl -X POST http://127.0.0.1:8000/identity/reload-known-faces
```

Example startup log:

```text
[INFO] Loaded 4 images for Rishi
[INFO] Registered identity: Rishi
[INFO] Loaded 4 images for Brother
[INFO] Registered identity: Brother
[WARNING] No face found in image sample.jpg
```

## Unknown Detection Behavior

- If a face is detected but does not meet the similarity threshold, HomeMindAI labels it as `UNKNOWN`.
- The system saves a cropped face image into `data/unknown_faces/`.
- An `UNKNOWN_PERSON_DETECTED` event is emitted with timestamp, camera ID, confidence, and tracking ID.

## Snapshot System

- Motion-triggered snapshots are stored automatically in `data/snapshots/`.
- Manual snapshots can be captured through `POST /camera/snapshot`.
- Unknown visitor face crops are stored automatically in `data/unknown_faces/`.
- Snapshot filenames include the camera ID, trigger reason, and a UTC timestamp.

## Camera Modes

### Webcam mode

```bash
cp .env.example .env
uvicorn main:app --reload
```

### RTSP mode

Update `.env`:

```env
HOMEMINDAI_CAMERA_SOURCE=rtsp
HOMEMINDAI_RTSP_URL=rtsp://username:password@ip:554/stream
```

Then run:

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/camera/live` to view the stream in a browser.

## Commands To Test Recognition

Start the service:

```bash
cd HomeMindAI
source .venv/bin/activate
uvicorn main:app --reload
```

Check system status:

```bash
curl http://127.0.0.1:8000/health
```

Reload known faces after adding images:

```bash
curl -X POST http://127.0.0.1:8000/identity/reload-known-faces
```

Inspect recent identity events:

```bash
curl http://127.0.0.1:8000/events/recent
```

View live stream in a browser:

```text
http://127.0.0.1:8000/camera/live
```

## Future Architecture Overview

Planned next layers for HomeMindAI:

1. Multi-class object detection beyond person-only inference.
2. Stronger tracking with DeepSORT or ByteTrack.
3. SQLAlchemy-backed event persistence and alert history.
4. Behavioral intelligence, anomaly reasoning, and risk scoring.
5. Dashboard endpoints, WebSocket streaming, and notification channels.

## Recommended Terminal Commands

```bash
cd HomeMindAI
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/identity/reload-known-faces
curl http://127.0.0.1:8000/events/recent
```