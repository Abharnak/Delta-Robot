#!/usr/bin/env python3
"""
YOLOv8 Object Detection (Webcam + Static Image)
- Press 'q' to quit webcam window
- Shows only detections with confidence >= 75%
- Displays color labels, coordinates, and bounding box dimensions
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ---------------- Configuration ----------------
MODEL_PATH = r"C:\Users\abhar\OneDrive\Desktop\Final_year_project\Object_detection\best.pt"  # Path to your trained weights
IMG_SIZE = 640          # Resize for inference
CONF_THRESHOLD = 0.75   # Show only predictions >= 75%
CAM_INDEX = 0           # 0 = laptop webcam
MM_PER_PIXEL = 0.26     # Conversion for real-world size estimation
# ------------------------------------------------


# ==========================================================
# Function: Draw bounding boxes and info on image
# ==========================================================
def draw_bounding_boxes(img, results, model):
    h, w, _ = img.shape
    banner_height = 30
    cv2.rectangle(img, (0, 0), (w, banner_height), (0, 0, 0), -1)
    resolution_text = f"Resolution: {w} x {h} pixels"
    cv2.putText(img, resolution_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue  # Skip low-confidence detections

        cls = int(box.cls[0])
        label = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        bw, bh = x2 - x1, y2 - y1

        # Convert to mm
        bw_mm, bh_mm = bw * MM_PER_PIXEL, bh * MM_PER_PIXEL
        cx_mm, cy_mm = round(cx * MM_PER_PIXEL,2), round(cy * MM_PER_PIXEL,2)

        # Print info to console
        print(f"Color: {label} | Conf: {conf:.2f} | Center: ({cx},{cy}) | Width: {bw} | Height: {bh}")

        # Draw rectangle and labels
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} ({conf*100:.1f}%)", (x1, y1 - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"W:{bw_mm:.1f}mm H:{bh_mm:.1f}mm", (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Center point + coordinates
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img, f"({cx},{cy})", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"({cx_mm},{cy_mm})", (cx + 10, cy+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img


# ==========================================================
# Function: Run on static image
# ==========================================================
def run_on_image(model, img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    results = model(img, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)[0]

    annotated = draw_bounding_boxes(img, results, model)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# ==========================================================
# Function: Run live webcam detection
# ==========================================================
def run_webcam(model):
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not access camera index {CAM_INDEX}")

    print("[INFO] Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        results = model(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)[0]
        annotated = draw_bounding_boxes(frame, results, model)

        cv2.imshow("YOLOv8 Live Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stream closed.")


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Choose mode:
    MODE = "webcam"   # "webcam" or "image"
    # MODE = "image"   # "webcam" or "image"

    if MODE == "webcam":
        run_webcam(model)
    else:
        # Path to your test image
        img_path = r"C:\Users\abhar\OneDrive\Desktop\Final_year_project\Object_detection\colored-blocks-9\test\images\IMG_4298_JPG.rf.1ba02f224dc582b1640b2f7c3deee6ae.jpg"
        run_on_image(model, img_path)
