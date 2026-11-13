# SignSight Backend

Flask backend for SignSight, the real-time sign language recognition web app. It serves a YOLOv8n detection model over HTTP so the React frontend can send images, videos and live camera frames and get annotated results back.

## How it works

- The YOLOv8n model (`weights/yolov8n.pt`) is loaded **once at startup**, not per request.
- Uploaded videos are resized to 640x480 and run frame by frame through the model.
- Every prediction uses a confidence threshold of **0.45**.
- Annotated results are saved under `static/results/` and served back over HTTP.
- Video processing runs in a background thread, so the upload request returns immediately.

## Endpoints

| Method | Endpoint | What it does |
| ------ | -------- | ------------ |
| GET | `/` | Health check, lists the available endpoints |
| POST | `/upload` | Upload an image or video file, returns annotated results |
| POST | `/detect_frame` | Send one base64 camera frame, get an annotated JPEG back |
| GET | `/static/results/<folder>/<filename>` | Download a processed image or video |

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The server starts at `http://127.0.0.1:5000/`. CORS is enabled, so the SignSight React frontend can call it directly from the browser.

## Stack

- Python, Flask 3 + flask-cors
- Ultralytics YOLOv8n, OpenCV, NumPy
- Gunicorn for production deploys

## Related repos

- Frontend: [signsight-frontend](https://github.com/immishra13/signsight-frontend)
- Combined platform: [AI-Powered-Dual-Vision-Intelligence-Platform](https://github.com/immishra13/AI-Powered-Dual-Vision-Intelligence-Platform)
