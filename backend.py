#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Model WebSocket Server: YOLOv11 + MediaPipe
- websockets v12+ compatible (single-arg handler)
- Best-of-two (YOLO vs MediaPipe) with >50% threshold
"""

import asyncio
import websockets
import json
import cv2
import numpy as np
import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime
import base64
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ------------------- CONFIG -------------------
WEIGHTS = Path(
    r"C:\Users\adity\Downloads\Sign-Language-Recognition-Using-Mediapipe-and-React-main\Sign-Language-Recognition-Using-Mediapipe-and-React-main\src\components\Detection\backend\best.pt"
)
IMG = 640
CONF = 0.35
IOU = 0.45
DEVICE = ""
MIN_COMMIT_CONF = 50.0     # strictly > 50
WEBSOCKET_PORT = 8765

STABLE_FRAMES = 3
COOLDOWN_FRAMES = 8
MAX_SENTENCE_CHARS = 220
HISTORY_MAX = 8
# ------------------------------------------------

def ensure_ultralytics():
    try:
        import ultralytics  # noqa
        from ultralytics import YOLO
        return YOLO
    except Exception:
        logger.info("Installing ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.50"])
        from ultralytics import YOLO
        return YOLO

def ensure_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except Exception:
        logger.info("Installing mediapipe...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.14"])
        import mediapipe as mp
        return mp

SPECIAL = {
    "space": " ", "Space": " ", "SPACE": " ", "blank": " ",
    "del": "<DEL>", "Del": "<DEL>", "DELETE": "<DEL>",
    "comma": ",", "period": ".", "dot": ".", "question": "?", "exclamation": "!",
}

def is_char_token(tok: str) -> bool:
    return isinstance(tok, str) and len(tok) == 1

def polish_grammar(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,!.?])", r"\1", s)
    s = re.sub(r"([,!.?])(?!\s|$)", r"\1 ", s)
    s = re.sub(r"\bi\b", "I", s)
    s = re.sub(r"(?i)\bi'm\b", "I'm", s)
    s = re.sub(r"(?i)\bi've\b", "I've", s)
    s = re.sub(r"(?i)\bi\'ve\b", "I've", s)
    s = re.sub(r"(?i)\bi'll\b", "I'll", s)
    def cap_after(m): return m.group(1) + m.group(2).upper()
    s = re.sub(r"(^|\.\s+|\!\s+|\?\s+)([a-z])", cap_after, s)
    return s.strip()

class DualModelDetector:
    def __init__(self):
        logger.info("🚀 Initializing Dual Model Detector...")

        # YOLO
        self.yolo_model = None
        try:
            YOLO = ensure_ultralytics()
            if WEIGHTS.exists():
                logger.info(f"📁 Loading YOLO model: {WEIGHTS}")
                self.yolo_model = YOLO(str(WEIGHTS))
                logger.info("✅ YOLO model loaded successfully!")
            else:
                logger.error(f"❌ YOLO weights not found at: {WEIGHTS}")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO: {e}")

        # MediaPipe Hands (for presence / placeholder gesture)
        self.mp_hands = None
        try:
            mp = ensure_mediapipe()
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("✅ MediaPipe initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")

        # sentence state
        self.sentence = ""
        self.prev_det_label = None
        self.stable = 0
        self.cooldown = 0
        self.last_committed_token = None
        self.last_type = None
        self.history = deque(maxlen=HISTORY_MAX)

        logger.info("🎯 Dual Model Detector ready!")

    def detect_yolo(self, frame):
        if not self.yolo_model:
            return {"model": "YOLO", "label": None, "confidence": 0.0, "detected": False}
        try:
            results = self.yolo_model.predict(
                source=frame, imgsz=IMG, conf=CONF, iou=IOU, device=DEVICE, verbose=False
            )
            r0 = results[0]
            names = r0.names if hasattr(r0, "names") else getattr(self.yolo_model.model, "names", {})

            if r0.boxes is not None and len(r0.boxes) > 0:
                confs = r0.boxes.conf.cpu().numpy()
                idx = int(np.argmax(confs))
                cls_id = int(r0.boxes.cls.cpu().numpy()[idx])
                conf = float(confs[idx]) * 100.0
                label = str(names.get(cls_id, f"class_{cls_id}"))
                return {"model": "YOLO", "label": label, "confidence": conf, "detected": conf > MIN_COMMIT_CONF}
            return {"model": "YOLO", "label": None, "confidence": 0.0, "detected": False}
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return {"model": "YOLO", "label": None, "confidence": 0.0, "detected": False}

    def detect_mediapipe(self, frame):
        if not self.mp_hands:
            return {"model": "MediaPipe", "label": None, "confidence": 0.0, "detected": False}
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)
            if results.multi_hand_landmarks:
                # yahan aap apna gesture label/score plug kar sakte ho
                return {"model": "MediaPipe", "label": "HAND_DETECTED", "confidence": 65.0, "detected": True}
            return {"model": "MediaPipe", "label": None, "confidence": 0.0, "detected": False}
        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
            return {"model": "MediaPipe", "label": None, "confidence": 0.0, "detected": False}

    def detect_dual(self, frame):
        yolo_result = self.detect_yolo(frame)
        mp_result = self.detect_mediapipe(frame)

        best = yolo_result if yolo_result["confidence"] >= mp_result["confidence"] else mp_result

        if best["detected"]:
            self.process_for_sentence(best["label"], best["confidence"])
            now_str = datetime.now().strftime("%I:%M:%S %p").lower()
            self.history.append((best["label"], best["confidence"], now_str))

        return {
            "best": best,
            "yolo": yolo_result,
            "mediapipe": mp_result,
            "timestamp": datetime.now().strftime("%I:%M:%S %p").lower(),
            "sentence": polish_grammar(self.sentence),
            "raw_sentence": self.sentence,
            "history": list(self.history),
        }

    def process_for_sentence(self, det_label, det_conf):
        if det_label is None or det_conf is None or det_conf <= MIN_COMMIT_CONF:
            return

        if det_label == self.prev_det_label:
            self.stable += 1
        else:
            self.prev_det_label = det_label
            self.stable = 1

        if self.cooldown > 0:
            self.cooldown -= 1

        if self.stable >= STABLE_FRAMES and self.cooldown == 0:
            token = SPECIAL.get(det_label, det_label)
            if token != self.last_committed_token:
                if token == "<DEL>":
                    self.sentence = self.sentence[:-1]
                    self.last_type = None
                elif token == " ":
                    if not self.sentence.endswith(" "):
                        self.sentence += " "
                    self.last_type = None
                elif token in [",", ".", "!", "?"]:
                    self.sentence = self.sentence.rstrip() + token + " "
                    self.last_type = None
                else:
                    if is_char_token(token):
                        if self.last_type == "word" and not self.sentence.endswith(" "):
                            self.sentence += " "
                        self.sentence += token
                        self.last_type = "char"
                    else:
                        if len(self.sentence) and not self.sentence.endswith(" "):
                            self.sentence += " "
                        self.sentence += token + " "
                        self.last_type = "word"

                if len(self.sentence) > MAX_SENTENCE_CHARS:
                    self.sentence = self.sentence[-MAX_SENTENCE_CHARS:]

                self.last_committed_token = token
                self.cooldown = COOLDOWN_FRAMES
                self.stable = 0

# Global state
detector = None
connected_clients = set()

# -------- websockets v12+ compatible handler (single-arg) --------
async def handle_client(websocket):
  """
  Handle WebSocket client (no 'path' arg in websockets>=12)
  """
  global detector
  client_addr = websocket.remote_address
  connected_clients.add(websocket)
  logger.info(f"🔗 Client connected: {client_addr}")

  try:
      # initial status
      await websocket.send(json.dumps({
          "type": "status",
          "message": "Server connected! Models ready.",
          "yolo_loaded": detector.yolo_model is not None,
          "mediapipe_loaded": detector.mp_hands is not None
      }))

      async for message in websocket:
          try:
              data = json.loads(message)

              if data.get("type") == "init":
                  resp = {
                      "type": "status",
                      "message": "Dual model detector ready!",
                      "models": ["YOLOv11", "MediaPipe"],
                      "yolo_loaded": detector.yolo_model is not None,
                      "mediapipe_loaded": detector.mp_hands is not None
                  }
                  await websocket.send(json.dumps(resp))
                  logger.info("✅ Sent init response")

              elif data.get("type") == "frame":
                  try:
                      image_data_str = data["image"]
                      if "," in image_data_str:
                          image_data_str = image_data_str.split(",")[1]
                      image_data = base64.b64decode(image_data_str)
                      nparr = np.frombuffer(image_data, np.uint8)
                      frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                      if frame is None:
                          raise ValueError("Failed to decode frame")

                      result = detector.detect_dual(frame)
                      await websocket.send(json.dumps({ "type": "detection", "result": result }))

                  except Exception as e:
                      logger.error(f"Frame processing error: {e}")
                      await websocket.send(json.dumps({ "type": "error", "message": f"Frame processing failed: {e}" }))

              elif data.get("type") == "reset":
                  detector.sentence = ""
                  detector.prev_det_label = None
                  detector.stable = 0
                  detector.cooldown = 0
                  detector.last_committed_token = None
                  detector.last_type = None
                  detector.history.clear()
                  await websocket.send(json.dumps({ "type": "status", "message": "Session reset successfully" }))

              else:
                  logger.warning(f"Unknown message type: {data.get('type')}")

          except json.JSONDecodeError:
              await websocket.send(json.dumps({ "type": "error", "message": "Invalid JSON format" }))
          except Exception as e:
              logger.error(f"Message processing error: {e}")
              await websocket.send(json.dumps({ "type": "error", "message": f"Processing error: {e}" }))

  except websockets.exceptions.ConnectionClosed:
      logger.info(f"❌ Client disconnected: {client_addr}")
  except Exception as e:
      logger.error(f"Connection error with {client_addr}: {e}")
  finally:
      connected_clients.discard(websocket)

async def main():
    global detector
    logger.info("🚀 Starting Dual Model Detection Server...")
    detector = DualModelDetector()

    if not detector.yolo_model and not detector.mp_hands:
        logger.error("❌ No models loaded successfully!")
        return

    logger.info(f"🌐 Starting WebSocket server on localhost:{WEBSOCKET_PORT}")

    server = await websockets.serve(
        handle_client,            # single-arg handler (v12+)
        "localhost",
        WEBSOCKET_PORT,
        ping_interval=30,
        ping_timeout=10,
        close_timeout=10,
    )

    logger.info("✅ Server ready! Waiting for React frontend connection...")
    logger.info(f"📡 WebSocket URL: ws://localhost:{WEBSOCKET_PORT}")
    logger.info(f"🎯 YOLO Model: {'✅ Loaded' if detector.yolo_model else '❌ Not Loaded'}")
    logger.info(f"🤖 MediaPipe: {'✅ Loaded' if detector.mp_hands else '❌ Not Loaded'}")

    await server.wait_closed()

if __name__ == "__main__":
    try:
        print("🚀 Starting Dual Model Detection Server...")
        print(f"📁 Model Path: {WEIGHTS}")
        print(f"📡 WebSocket Server: ws://localhost:{WEBSOCKET_PORT}")
        print("Press Ctrl+C to stop")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")
    except Exception as e:
        print(f"❌ Server error: {e}")
