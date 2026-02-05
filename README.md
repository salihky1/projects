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

---

## 1️⃣ Smart Home Device Event Prediction API

Provides predictive control for home devices including water pumps, AC units, and face recognition.

### Endpoints

#### Generate Random Events
POST /generate_data?count=100
Generates random sensor/face/manual events and stores them in Firebase.

#### Train Predictive Models
POST /train_models
Trains machine learning models for motor, AC, and face prediction based on historical events.

#### Motor Alert
POST /alert/motor
Returns motor ON/OFF suggestion based on soil moisture, temperature, and humidity. Includes confidence score.

#### AC Alert
POST /alert/ac
Returns AC ON/OFF suggestion based on room temperature and humidity. Includes confidence score.

#### Face Hourly Prediction
POST /predict/face_hourly
Predicts the most likely person to enter the house at a given hour. Returns confidence level.

#### Save Feedback
POST /feedback
Records user feedback for predictions.

#### Event History
GET /events/history?device_id=home_device_1&limit=50
Retrieves last X events, optionally filtered by device.

---

## 2️⃣ Contour & Convex Hull (`contour_convex_hull.py`)

Detects contours in an image, computes convex hulls, and visualizes them.

**Usage:**
```python
from contour_convex_hull import process_image
process_image("yildiz.jpg")
3️⃣ Sharpest Cutout (sharpest_cutout.py)
Removes image background using the rembg library and U²-Net model.
Usage:

from sharpest_cutout import SharpestCutout
remover = SharpestCutout("input_image.png")
remover.show()
remover.save("output.png")
4️⃣ Background Subtractor (background_subtractor.py)
Performs background subtraction on video files using OpenCV’s MOG2 algorithm.
Usage:

from background_subtractor import BackgroundSubtractor
video_path = "video.mp4"
bg_subtractor = BackgroundSubtractor(video_path)
bg_subtractor.run()
⚡ Requirements
Install dependencies using pip:
pip install -r requirements.txt
Example requirements.txt:
fastapi
uvicorn
pandas
numpy
scikit-learn
opencv-python
Pillow
rembg
firebase-admin
joblib
▶️ Running the FastAPI API
Start the API:
uvicorn smart_home_device_event_prediction_api:app --reload
The API will be available at: http://127.0.0.1:8000
Interactive documentation is available at: http://127.0.0.1:8000/docs
💡 Notes
The face recognition model predicts likely visitors based on hour of day.
Motor and AC predictions include confidence scores; low confidence is flagged.
Background removal produces sharp edges without alpha matting blur.
Contour & convex hull detection visualizes shapes in blue (contours) and green (hulls).
🛠 Technologies
Python 3.x
FastAPI
OpenCV
NumPy
pandas
scikit-learn
Pillow
rembg (U²-Net)
Firebase Firestore
📚 Future Improvements
Add real-time sensor streaming support.
Expand predictive models with additional features (light, motion sensors).
Enhance face recognition with deep learning embeddings.
Add user authentication for API access.
