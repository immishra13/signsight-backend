# app.py (Backend Logic Only)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import os
import threading
import base64
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
# Use the model from the first code snippet
MODEL_PATH = Path("weights/yolov8n.pt") 
RESULT_FOLDER_NAME = "static/results"
RESULT_FOLDER = Path(RESULT_FOLDER_NAME)

# Frame dimensions for video upload processing (from original code)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONF_THRESHOLD = 0.45 

# Create results directory if it doesn't exist
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the YOLO model once
try:
    # Load YOLOv8n with the correct task type "v8"
    model = YOLO(MODEL_PATH, "v8") 
    print(f"YOLO Model (v8n) loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model from {MODEL_PATH}: {e}")
    print("Please ensure 'weights/yolov8n.pt' exists in the 'weights' directory.")
    exit()

# --- FLASK APP SETUP ---
app = Flask(__name__, static_url_path='/static')
CORS(app) # Enables cross-origin requests

# --- UTILITY FUNCTIONS (Core Processing Logic) ---

def process_video(video_path: str, save_dir: str):
    """Run YOLO on a full video and save annotated mp4. Resizes to 640x480."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    
    width = FRAME_WIDTH
    height = FRAME_HEIGHT

    out_path = Path(save_dir) / "detected_output.mp4"
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"Processing video: {video_path}")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
            
        # Resize frame to the standard size (640x480)
        frame = cv2.resize(frame, (width, height)) 
        
        # Run inference with the configured threshold
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False) 
        
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()
    print(f"Video processing finished. Output saved to {out_path}")


def process_image(image_path: str, save_dir: str):
    """Run YOLO on a single image and save annotated jpg."""
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image file {image_path}")
        return

    # Run model inference with the configured threshold
    results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()
    out_path = Path(save_dir) / "detected_output.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"Image processing finished. Output saved to {out_path}")


# --- FLASK ROUTES (API Endpoints) ---

@app.route("/", methods=["GET"])
def home_info():
    """Provides basic info and endpoint guidance."""
    return jsonify({
        "message": "YOLOv8n Detection Backend Running",
        "endpoints": {
            "/upload (POST)": "Handles image and video file uploads for detection.",
            "/detect_frame (POST)": "Handles real-time base64 image streams (e.g., from webcam).",
            f"/{RESULT_FOLDER_NAME}/<folder>/<filename> (GET)": "Serves detection results.",
        }
    })

@app.route("/upload", methods=["POST"])
def upload_file():
    """Image/Video upload + detection endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    filename = f.filename
    if not filename:
         return jsonify({"error": "No selected file"}), 400

    ext = filename.rsplit(".", 1)[-1].lower()
    base_name = os.path.splitext(filename)[0]
    
    out_dir = RESULT_FOLDER / base_name
    os.makedirs(out_dir, exist_ok=True)
    
    saved_path = out_dir / filename
    f.save(str(saved_path))
    
    output_url_prefix = f"/{RESULT_FOLDER_NAME}/{base_name}"

    if ext in {"mp4", "avi", "mov", "mkv"}:
        # Process video in a non-blocking background thread
        thread = threading.Thread(target=process_video, args=(str(saved_path), str(out_dir)))
        thread.start()

        return jsonify(
            {
                "success": True,
                "type": "video",
                "message": "Video uploaded. Processing started in background. The detected video will be available shortly.",
                "original_video": f"{output_url_prefix}/{filename}", 
                "detected_video_url": f"{output_url_prefix}/detected_output.mp4",
            }
        )

    if ext in {"jpg", "jpeg", "png", "bmp"}:
        # Process image synchronously
        process_image(str(saved_path), str(out_dir))
        
        return jsonify(
            {
                "success": True,
                "type": "image",
                "detected_image_url": f"{output_url_prefix}/detected_output.jpg",
            }
        )

    return jsonify({"error": f"Unsupported file format: .{ext}"}), 400


@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    """
    Live detection endpoint: accepts base64 image, runs YOLO, and returns 
    annotated base64 image.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"ok": False, "error": "No image payload"}), 400

    data_url = data["image"]
    if "," in data_url:
        data_url = data_url.split(",")[1]
        
    try:
        # Decode base64 to OpenCV image (numpy array)
        img_bytes = base64.b64decode(data_url)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to decode image: {e}"}), 500

    # Run model inference
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False) 
    annotated = results[0].plot()

    # Encode annotated image back to base64
    _, enc = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(enc.tobytes()).decode("utf-8")
    
    return jsonify({"ok": True, "image": f"data:image/jpeg;base64,{b64}"})


@app.route(f"/{RESULT_FOLDER_NAME}/<folder>/<filename>")
def serve_results(folder, filename):
    """Endpoint to serve processed image/video files."""
    return send_from_directory(str(RESULT_FOLDER / folder), filename)


if __name__ == "__main__":
    print("-" * 50)
    print("FLASK YOLOv8n BACKEND STARTED")
    print(f"API Base URL: http://127.0.0.1:5000/")
    print("-" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)