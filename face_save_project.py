from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from deepface import DeepFace
from PIL import Image
import numpy as np
import io
import os
import cv2
import json

app = FastAPI(title="Face Recognition API", version="1.0")

# -------------------------------
# Directory for storing embeddings
# -------------------------------
SAVE_DIR = "user_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------
# Head directions for registration
# -------------------------------
directions = ["Turn Right", "Turn Left", "Look Up", "Look Down", "Look Center"]

current_name = None
current_embeddings = []
current_step = 0


# =========================================================
# 1) START FACE REGISTRATION
# =========================================================
@app.post("/face/register/start")
async def start_registration(name: str = Form(...)):
    """
    Starts the face registration process for a user.
    """
    global current_name, current_embeddings, current_step

    current_name = name.strip()
    current_embeddings = []
    current_step = 0

    return {
        "status": "ok",
        "message": f"Registration started for {current_name}",
        "next_step": directions[current_step]
    }


# =========================================================
# 2) REGISTER EACH STEP (IMAGE UPLOAD)
# =========================================================
@app.post("/face/register/step")
async def register_step(image: UploadFile = File(...)):
    """
    Receives an image for the current direction and extracts embeddings.
    """
    global current_step, current_embeddings

    image_bytes = await image.read()
    img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    try:
        embedding_result = DeepFace.represent(
            img,
            model_name="Facenet",
            enforce_detection=True
        )

        current_embeddings.append(embedding_result[0]["embedding"])
        message = f"{directions[current_step]} step completed"
        current_step += 1

    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": f"Face could not be detected: {str(e)}"},
            status_code=400
        )

    if current_step >= len(directions):
        file_path = os.path.join(SAVE_DIR, f"{current_name}_faces.npy")
        np.save(file_path, current_embeddings)
        return {"status": "done", "message": "All steps completed successfully"}

    return {
        "status": "ok",
        "message": message,
        "next_step": directions[current_step]
    }


# =========================================================
# 3) FINISH REGISTRATION
# =========================================================
@app.post("/face/register/finish")
async def finish_registration():
    """
    Saves all collected embeddings to disk.
    """
    global current_name, current_embeddings

    file_path = os.path.join(SAVE_DIR, f"{current_name}_faces.npy")
    np.save(file_path, current_embeddings)

    return {
        "status": "ok",
        "message": f"Registration completed for {current_name}"
    }


# =========================================================
# COSINE SIMILARITY FUNCTION
# =========================================================
def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)


# =========================================================
# 4) FACE RECOGNITION ENDPOINT
# =========================================================
@app.post("/face/recognize")
async def recognize_face(
    embeddings_json: str = Form(...),   # Sent from Flutter
    image: UploadFile = File(...)
):
    """
    Recognizes a face by comparing it with embeddings sent from client.
    """
    try:
        # -------------------------------
        # Read Image
        # -------------------------------
        img_bytes = await image.read()
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------------------
        # Parse embeddings from Flutter
        # Format: [{"name": "John", "embedding": [...]}, ...]
        # -------------------------------
        embeddings_list = json.loads(embeddings_json)
        names_list = [e["name"] for e in embeddings_list]
        vectors_list = [np.array(e["embedding"]) for e in embeddings_list]

        # -------------------------------
        # Extract embedding from image
        # -------------------------------
        result = DeepFace.represent(
            rgb_frame,
            model_name="Facenet512"
        )
        input_embedding = result[0]["embedding"]

        # -------------------------------
        # Compare embeddings
        # -------------------------------
        best_score = 0.0
        best_name = None

        for i, vec in enumerate(vectors_list):
            similarity = cosine_similarity(input_embedding, vec)
            if similarity > best_score:
                best_score = similarity
                best_name = names_list[i]

        # -------------------------------
        # Return Result
        # -------------------------------
        if best_score >= 0.65:
            return JSONResponse({
                "status": "ok",
                "recognized_name": best_name,
                "similarity": best_score,
                "message": f"Recognized: {best_name} ({best_score*100:.1f}%)"
            })
        else:
            return JSONResponse({
                "status": "ok",
                "recognized_name": None,
                "similarity": best_score,
                "message": f"Not recognized ({best_score*100:.1f}%)"
            })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})
