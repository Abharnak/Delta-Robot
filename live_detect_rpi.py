from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import argparse
import os

# --- Parse args for headless mode ---
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args = parser.parse_args()
HEADLESS = args.headless

# --- Load YOLOv8 model ---
model_path = "/home/pi/Desktop/Object_detection/model/best.pt"
model = YOLO(model_path)
print(f"[INFO] Loaded YOLOv8 model from {model_path}")
print("[INFO] Press 'q' to quit.")

# --- Initialize Picamera2 ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

try:
    while True:
        # Capture frame from Picamera2 (RGB)
        frame = picam2.capture_array()

        # Convert RGB -> BGR for OpenCV / YOLOv8 compatibility
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run YOLO inference
        results = model(frame_bgr, conf=0.5)

        # Annotate frame
        annotated_frame = results[0].plot()  # returns BGR image

        if not HEADLESS:
            cv2.imshow("YOLOv8 RPi Camera", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Headless mode: save frame to disk (optional)
            os.makedirs("output", exist_ok=True)
            cv2.imwrite("output/frame.jpg", annotated_frame)

except KeyboardInterrupt:
    print("\n[INFO] Exiting program.")

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()  # only call if GUI available
    print("\n[INFO] Stream closed.")
