#!/usr/bin/env python3
"""
YOLOv8 live object detection using laptop webcam
Press 'q' to quit the window
"""

from ultralytics import YOLO
import cv2

# ---------------- Configuration ----------------
MODEL_PATH = r"runs\detect\yolov8_colored_blocks2\weights\best.pt"  # Path to your trained weights
IMG_SIZE = 640          # Resize for inference speed/accuracy
CAM_INDEX = 0           # 0 = default webcam, 1 = external
# ------------------------------------------------

def main():
    # Load YOLOv8 model
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Open webcam
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not access camera index {CAM_INDEX}")

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        # Run detection
        results = model(frame, imgsz=IMG_SIZE, verbose=False)

        # Annotate frame with bounding boxes & labels
        annotated = results[0].plot()

        # Show window
        cv2.imshow("YOLOv8 Live Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed.")

if __name__ == "__main__":
    main()



"""Test Black Screen"""
# import cv2
# import numpy as np

# img = np.zeros((200,200,3), dtype=np.uint8)
# cv2.imshow("Test Window", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
