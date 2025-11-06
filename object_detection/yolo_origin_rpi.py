#!/usr/bin/env python3
"""
YOLOv8 Live Detection with Origin and Scale Calibration (Raspberry Pi)
- Click 2 points with known distance (e.g., 50 mm) to auto-calculate scale
- Click 1 point to set origin
- Displays coordinates in mm relative to origin (+X right, +Y up)
- Press 'q' to quit
"""

from ultralytics import YOLO
import cv2
import numpy as np
import math
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "/home/pi/best.pt"   # 👈 change to your model path
IMG_SIZE = 416                   # smaller for faster inference on Pi
CONF_THRESHOLD = 0.65
CAM_INDEX = 0
KNOWN_DISTANCE_MM = 100.0
# ----------------------------------------

# Globals
points = []
MM_PER_PIXEL = None
origin = None
calibrated = False
set_origin = False


# ---------------- Mouse callback ----------------
def mouse_callback(event, x, y, flags, param):
    global points, MM_PER_PIXEL, origin, calibrated, set_origin

    if event == cv2.EVENT_LBUTTONDOWN:
        if not calibrated:
            points.append((x, y))
            print(f"[CLICK] Calibration point {len(points)}: ({x},{y})")
            if len(points) == 2:
                dist_px = math.dist(points[0], points[1])
                MM_PER_PIXEL = KNOWN_DISTANCE_MM / dist_px
                calibrated = True
                print(f"[INFO] ✅ Scale set: 1 px = {MM_PER_PIXEL:.4f} mm")
                print("Now click a point to set the ORIGIN.")
        elif not set_origin:
            origin = (x, y)
            set_origin = True
            print(f"[INFO] ✅ Origin set at: {origin}")


# ---------------- Draw bounding boxes ----------------
def draw_bounding_boxes(img, results, model):
    global MM_PER_PIXEL, origin

    if not (MM_PER_PIXEL and origin):
        return img

    h, w, _ = img.shape
    # Draw axes
    cv2.line(img, (0, origin[1]), (w, origin[1]), (255, 0, 0), 2)  # X-axis
    cv2.line(img, (origin[0], 0), (origin[0], h), (0, 255, 0), 2)  # Y-axis
    cv2.circle(img, origin, 5, (0, 0, 255), -1)
    cv2.putText(img, "Origin", (origin[0] + 10, origin[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls = int(box.cls[0])
        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        bw, bh = x2 - x1, y2 - y1

        # Convert pixel to mm relative to origin (+Y upward)
        x_mm = (cx - origin[0]) * MM_PER_PIXEL
        y_mm = -(cy - origin[1]) * MM_PER_PIXEL
        bw_mm, bh_mm = bw * MM_PER_PIXEL, bh * MM_PER_PIXEL

        # Draw box and info
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img, f"{label} ({conf*100:.1f}%)", (x1, y1 - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"W:{bw_mm:.1f}mm H:{bh_mm:.1f}mm", (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, f"({x_mm:.1f},{y_mm:.1f})mm", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f"{label}: ({x_mm:.2f}, {y_mm:.2f}) mm")

    return img


# ---------------- Webcam loop ----------------
def run_webcam(model):
    global calibrated, set_origin
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not access camera index {CAM_INDEX}")

    cv2.namedWindow("YOLOv8 Detection")
    cv2.setMouseCallback("YOLOv8 Detection", mouse_callback)

    print("[INFO] Click 2 points (50 mm apart) for calibration, then click origin.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        if calibrated and set_origin:
            results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)[0]
            frame = draw_bounding_boxes(frame, results, model)
        else:
            msg = "Click 2 pts (50mm apart)" if not calibrated else "Click origin"
            cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # FPS display
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stream closed.")


# ---------------- Main ----------------
if __name__ == "__main__":
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    run_webcam(model)
