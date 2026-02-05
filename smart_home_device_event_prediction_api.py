from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import uuid, os, joblib, random, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import firebase_admin
from firebase_admin import credentials, firestore

# Model paths
MODEL_MOTOR = "model_motor.joblib"
MODEL_AC = "model_ac.joblib"
MODEL_FACE = "model_face.joblib"

# Firebase initialization
if not firebase_admin._apps:
    cred = credentials.Certificate("goruntuIsleme.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

# FastAPI app
app = FastAPI(title="Smart Home Device Event Prediction API", version="1.2.0")


class Event(BaseModel):
    device_id: str
    timestamp: Optional[str] = None
    event_type: str
    identity: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    soilMoisture: Optional[float] = None
    ac_state: Optional[Dict[str, Any]] = None
    relay1: Optional[bool] = None
    relay2: Optional[bool] = None


class Feedback(BaseModel):
    prediction_id: str
    accepted: bool


# Event message generator
def generate_message(device, sensor_val, action, timestamp, identity=None):
    ts = pd.to_datetime(timestamp).strftime("%H:%M")
    if device == "motor":
        return f"Water pump {'should be turned ON' if action == 1 else 'can remain OFF'}? Soil moisture {sensor_val}"
    elif device == "ac":
        return f"Room temperature {sensor_val['temp']}°C, humidity {sensor_val['hum']}%. AC {'should turn ON' if action == 1 else 'can remain OFF'}."
    elif device == "camera" and identity:
        return f"{identity} likely entered home around {ts}."
    return ""


# Random event generator
def generate_random_event():
    now = datetime.utcnow()
    event_type = random.choice(["sensor", "face_detect", "manual"])
    identity = random.choice(["Salih", "Sevda", None]) if event_type == "face_detect" else None
    temp = round(random.uniform(18, 35), 2)
    humidity = round(random.uniform(20, 90), 2)
    soil = round(random.uniform(10, 80), 2)
    relay = 1 if (soil < 30 or humidity < 40) else 0
    ac_on = 1 if (temp > 28 or humidity > 70) else 0
    return {
        "id": str(uuid.uuid4()),
        "timestamp": now.isoformat(),
        "device_id": "home_device_1",
        "event_type": event_type,
        "identity": identity,
        "temperature": temp,
        "humidity": humidity,
        "soilMoisture": soil,
        "relay1": relay,
        "relay2": 0,
        "ac_on": ac_on,
        "ac_temp": temp if ac_on else None,
        "action_taken": "motor_on" if relay == 1 else "none",
        "feedback": None,
    }


# Generate multiple events
@app.post("/generate_data")
def generate_data(count: int = 100):
    batch = db.batch()
    for _ in range(count):
        ev = generate_random_event()
        ref = db.collection("events").document(ev["id"])
        batch.set(ref, ev)
    batch.commit()
    return {"status": "ok", "generated": count}


# Train predictive models
@app.post("/train_models")
def train_models():
    docs = db.collection("events").stream()
    rows = [d.to_dict() for d in docs]
    if not rows:
        raise HTTPException(400, "No data available to train models.")

    df = pd.DataFrame(rows)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df = df.fillna(0)

    # Motor model
    X_motor = df[['hour', 'temperature', 'humidity', 'soilMoisture']]
    y_motor = df['relay1'].astype(int)
    model_motor = RandomForestClassifier(n_estimators=100, random_state=42)
    model_motor.fit(X_motor, y_motor)
    joblib.dump(model_motor, MODEL_MOTOR)

    # AC model
    X_ac = df[['hour', 'temperature', 'humidity']]
    y_ac = df['ac_on'].astype(int)
    model_ac = RandomForestClassifier(n_estimators=100, random_state=42)
    model_ac.fit(X_ac, y_ac)
    joblib.dump(model_ac, MODEL_AC)

    # Face recognition model
    face_data = df[df['identity'].notnull()].copy()
    if len(face_data) > 5:
        face_data['identity'] = face_data['identity'].astype(str)
        X_face = face_data[['hour']]
        y_face = face_data['identity']
        model_face = RandomForestClassifier(n_estimators=100, random_state=42)
        model_face.fit(X_face, y_face)
        joblib.dump(model_face, MODEL_FACE)

    return {
        "status": "trained",
        "motor_data_count": len(X_motor),
        "ac_data_count": len(X_ac),
        "face_data_count": len(face_data)
    }


# Motor alert
@app.post("/alert/motor")
def motor_alert(payload: dict, confidence_threshold: float = 0.5):
    if not os.path.exists(MODEL_MOTOR):
        raise HTTPException(400, "Motor model not trained yet.")

    ts = payload.get("timestamp") or datetime.utcnow().isoformat()
    hour = pd.to_datetime(ts).hour
    soil = payload.get("soilMoisture", 0)
    temp = payload.get("temperature", 0)
    hum = payload.get("humidity", 0)

    model_motor = joblib.load(MODEL_MOTOR)
    X_motor = np.array([[hour, temp, hum, soil]])
    motor_pred = int(model_motor.predict(X_motor)[0])
    motor_prob = float(max(model_motor.predict_proba(X_motor)[0]))

    action_msg = "TURN ON" if motor_pred == 1 else "STAY OFF"
    if motor_prob < confidence_threshold:
        action_msg += " (low confidence)"

    msg = f"Soil moisture {soil}, motor suggested action: {action_msg}"
    return {"motor_action": motor_pred, "probability": motor_prob, "message": msg}


# AC alert
@app.post("/alert/ac")
def ac_alert(payload: dict, confidence_threshold: float = 0.5):
    if not os.path.exists(MODEL_AC):
        raise HTTPException(400, "AC model not trained yet.")

    ts = payload.get("timestamp") or datetime.utcnow().isoformat()
    hour = pd.to_datetime(ts).hour
    temp = payload.get("temperature", 0)
    hum = payload.get("humidity", 0)

    model_ac = joblib.load(MODEL_AC)
    X_ac = np.array([[hour, temp, hum]])
    ac_pred = int(model_ac.predict(X_ac)[0])
    ac_prob = float(max(model_ac.predict_proba(X_ac)[0]))

    action_msg = "TURN ON" if ac_pred == 1 else "STAY OFF"
    if ac_prob < confidence_threshold:
        action_msg += " (low confidence)"

    msg = f"Room temp {temp}°C, humidity {hum}%. AC suggested action: {action_msg}"
    return {"ac_action": ac_pred, "probability": ac_prob, "message": msg}


# Face hourly prediction
@app.post("/predict/face_hourly")
def face_hourly_predict(payload: dict, confidence_threshold: float = 0.5):
    if not os.path.exists(MODEL_FACE):
        raise HTTPException(400, "Face model not trained yet.")

    ts = payload.get("timestamp") or datetime.utcnow().isoformat()
    hour = pd.to_datetime(ts).hour

    model_face = joblib.load(MODEL_FACE)
    X_face = np.array([[hour]])
    identity_pred = model_face.predict(X_face)[0]
    confidence = float(max(model_face.predict_proba(X_face)[0]))

    msg = f"Likely person around {hour}:00 is {identity_pred} (confidence: {confidence:.2f})"
    if confidence < confidence_threshold:
        msg += " (low confidence)"

    return {"likely_person": identity_pred, "confidence": confidence, "message": msg}


# Feedback saving
@app.post("/feedback")
def save_feedback(f: Feedback):
    db.collection("events").document(f.prediction_id).update({
        "feedback": 1 if f.accepted else 0
    })
    return {"status": "feedback_saved"}


# Event history retrieval
@app.get("/events/history")
def get_event_history(device_id: Optional[str] = None, limit: int = 50):
    query = db.collection("events").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
    if device_id:
        query = query.where("device_id", "==", device_id)
    docs = query.stream()
    events = [doc.to_dict() for doc in docs]
    return {"events": events, "count": len(events)}
