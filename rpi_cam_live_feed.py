from picamera2 import Picamera2, Preview
import time

# Initialize camera
picam2 = Picamera2()

# Configure preview
config = picam2.create_preview_configuration()
picam2.configure(config)

# Start preview
picam2.start_preview(Preview.QTGL)  # opens GPU-accelerated live window
picam2.start()

print("📷 Live camera feed started. Press Ctrl+C to exit.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n👋 Exiting...")
    picam2.stop()
