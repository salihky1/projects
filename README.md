# Smart Home Device Event Prediction & Computer Vision Project

This repository contains multiple Python modules for smart home automation, device control prediction, computer vision, and background segmentation. It provides APIs for predictive control of devices, face detection, random event generation, and image/video processing.

---

## 📂 Project Structure

projects/
│
├── smart_home_device_event_prediction_api.py # Main FastAPI module for smart home predictions
├── contour_convex_hull.py # Detects contours and convex hulls on images
├── sharpest_cutout.py # Removes background from images using rembg
├── background_subtractor.py # Video background subtraction
├── requirements.txt
└── README.md

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

## 2️⃣ Contour & Convex Hull (`contour_convex_hull.py`)

Detects contours in an image, computes convex hulls, and visualizes them.

**Usage:**
```python
from contour_convex_hull import process_image
process_image("yildiz.jpg")
