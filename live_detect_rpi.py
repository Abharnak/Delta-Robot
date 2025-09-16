#!/usr/bin/env python3
"""
YOLOv8 live object detection using Raspberry Pi camera
Press 'q' to quit
Optimized for CPU
"""

from ultralytics import YOLO
import cv2
import time

# ---------------- Configuration ----------------
MODEL_PATH = "/home/pi/best.pt"  # Path to your trained YOLOv8 weights
IMG_SIZE = 320                   # Reduce for faster inference on CPU
CAM_INDEX = 0                     # 0 = Pi Camera (via libcamera) or USB cam
# ------------------------------------------------

def main():
    print(f"[INFO] Loading YOLOv8 model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Open Pi Camera / USB camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not access camera index {CAM_INDEX}")

    print("[INFO] Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        # Run YOLO inference (faster with smaller image size)
        results = model(frame, imgsz=IMG_SIZE, verbose=False)

        # Annotate frame with bounding boxes & labels
        annotated = results[0].plot()

        # Show window (Pi GUI required)
        cv2.imshow("YOLOv8 Live Detection", annotated)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        print(f"[INFO] FPS: {fps:.2f}", end='\r')

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Stream closed.")

if __name__ == "__main__":
    main()
