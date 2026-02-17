import cv2
import numpy as np
import time
import os
from datetime import datetime

# -----------------------------
# SETTINGS
# -----------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 800
FRAME_HEIGHT = 600
RECORD_VIDEO = True
SAVE_LOG = True

OUTPUT_DIR = "records"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# -----------------------------
# Load Haar Cascades
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# -----------------------------
# Open Camera
# -----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# -----------------------------
# Video Writer
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(OUTPUT_DIR, f"eye_record_{timestamp}.avi")
log_path = os.path.join(OUTPUT_DIR, f"eye_log_{timestamp}.txt")

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(video_path, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))

# -----------------------------
# Log file
# -----------------------------
if SAVE_LOG:
    log_file = open(log_path, "w")
    log_file.write("Time,X,Y,Width,Height\n")

# -----------------------------
# FPS Calculation
# -----------------------------
prev_time = 0

print("Eye Tracking Started")
print("Press Q to Quit | R to Start/Stop Recording")

recording = True

# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read error")
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0,255,0), 2)

        face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
        face_roi_color = frame[fy:fy+fh, fx:fx+fw]

        eyes = eye_cascade.detectMultiScale(face_roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_center_x = fx + ex + ew // 2
            eye_center_y = fy + ey + eh // 2

            # Draw rectangle around eye
            cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (255,0,0), 2)

            # Draw crosshair
            cv2.line(frame, (eye_center_x, 0), (eye_center_x, FRAME_HEIGHT), (255,255,0), 1)
            cv2.line(frame, (0, eye_center_y), (FRAME_WIDTH, eye_center_y), (255,255,0), 1)

            # Log data
            if SAVE_LOG and recording:
                current_time = time.time()
                log_file.write(f"{current_time},{eye_center_x},{eye_center_y},{ew},{eh}\n")

    # FPS calculation
    current_time = time.time()
    fps = int(1 / (current_time - prev_time)) if current_time != prev_time else 0
    prev_time = current_time

    # UI Panel
    cv2.rectangle(frame, (0,0), (FRAME_WIDTH, 60), (50,50,50), -1)
    cv2.putText(frame, "Eye Tracking & Recording System", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    status_text = "RECORDING" if recording else "PAUSED"
    cv2.putText(frame, f"Status: {status_text}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f"FPS: {fps}", (FRAME_WIDTH-120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Show
    cv2.imshow("Eye Tracking", frame)

    # Save video
    if RECORD_VIDEO and recording:
        out.write(frame)

    # Keyboard
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Program ended.")
        break

    if key == ord("r"):
        recording = not recording
        print("Recording:", recording)

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()

if SAVE_LOG:
    log_file.close()

cv2.destroyAllWindows()

print("Saved video:", video_path)
print("Saved log:", log_path)
