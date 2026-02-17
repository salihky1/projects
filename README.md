# Smart Home Device Event Prediction & Computer Vision Project

This repository contains multiple Python modules for smart home automation, device control prediction, computer vision, and background segmentation. It provides APIs for predictive control of devices, face detection, random event generation, and image/video processing.

---


## 1️⃣ Smart Home Device Event Prediction API

Provides predictive control for home devices including water pumps, AC units, and face recognition.

### Endpoints

- **Generate Random Events**  
  `POST /generate_data?count=100`  
  Generates random sensor/face/manual events and stores them in Firebase.

- **Train Predictive Models**  
  `POST /train_models`  
  Trains machine learning models for motor, AC, and face prediction based on historical events.

- **Motor Alert**  
  `POST /alert/motor`  
  Returns motor ON/OFF suggestion based on soil moisture, temperature, and humidity. Includes confidence score.

- **AC Alert**  
  `POST /alert/ac`  
  Returns AC ON/OFF suggestion based on room temperature and humidity. Includes confidence score.

- **Face Hourly Prediction**  
  `POST /predict/face_hourly`  
  Predicts the most likely person to enter the house at a given hour. Returns confidence level.

- **Save Feedback**  
  `POST /feedback`  
  Records user feedback for predictions.

- **Event History**  
  `GET /events/history?device_id=home_device_1&limit=50`  
  Retrieves last X events, optionally filtered by device.

---

## 2️⃣ Iridescent Metallic Cubes & Blender Material Inspector
Provides automatic inspection of Blender’s Principled BSDF shader inputs.
Features
Create Material
Automatically creates a new material and enables node-based shading.
Find Principled BSDF Node
Searches for the Principled BSDF shader node inside the material node tree.
List Shader Inputs
Prints all available input sockets (Base Color, Metallic, Roughness, etc.) to the console.
Automation Ready
Can be extended for procedural material creation and shader scripting.
2️⃣ Iridescent Metallic Cubes (Pygame)
Renders animated metallic cubes with iridescent color transitions using Pygame.
Features
3D Cube Projection
Draws pseudo-3D cubes using perspective projection.
Iridescent Color Generation
Uses sine wave functions to smoothly transition between blue, purple, turquoise, yellow, and orange.
Metallic Highlight Effect
Adds transparent faces and white edge highlights to simulate glossy metal.
Multiple Rotating Cubes
Displays four independently rotating cubes with different rotation speeds.
Background Grid
Draws a grid to enhance depth perception.
Keyboard Control
ESC key closes the application.

# 👁 Project 3: Eye Tracking & Recording System

# 📖 Description

This project is an eye tracking and recording system built using OpenCV and Python.  
It captures live video from a webcam, detects faces and eyes in real time, and calculates the center of the detected eye region.  
The system draws crosshair lines on the screen to visualize eye position and records both video and eye coordinates.

The project also includes:

Real-time face and eye detection  
Eye center calculation  
Video recording  
Coordinate logging to file  
FPS display  
Keyboard control system  

This project is useful for learning:

Face and eye detection using Haar cascades  
Contour and region processing  
Real-time video analysis  
Object tracking fundamentals  
Computer vision based interaction  

# ✨ Features

Live camera capture  
Face detection  
Eye detection  
Crosshair visualization  
Eye coordinate logging  
Video recording system  
FPS counter  
Start / stop recording  
On-screen UI panel  

# 🛠 Technologies

Python 3  
OpenCV  
NumPy  

# ▶ How to Run
1. Install dependencies  
pip install opencv-python numpy  

2. Run the script  
python eye_tracking.py  

# 🎮 Controls

Q → Quit program  
R → Start / Stop recording  

# 📂 Output Files

Recorded video file (.avi)  
Eye coordinate log file (.txt or .csv)  

Each session creates:

A new video file with timestamp  
A new log file with eye position data  

# 🚀 Future Improvements

Pupil detection  
Blink (eye closing) detection  
Gaze direction estimation  
Heatmap generation  
Mouse control using eyes  
GUI version with PyQt  
CSV export support  
Multiple face tracking  

