from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN
import cv2
from io import BytesIO

# Inisialisasi aplikasi dan limiter
app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Handler untuk pelanggaran rate limit
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Rate limit exceeded. Please try again later.", status_code=429)

# Model TensorFlow
model = None
emotion_labels = ["Angry", "Fear", "Happy", "Neutral", "Sad"]

@app.on_event("startup")
async def load_model():
    global model
    model = tf.keras.models.load_model("model/expression_detection_new.h5")

# Inisialisasi MTCNN
detector = MTCNN(keep_all=True)

@app.post("/detect-expression/")
@limiter.limit("10/minute")
async def detect_expression(request: Request, file: UploadFile = File(...)):
    """
    Endpoint untuk mendeteksi ekspresi wajah dari gambar.
    """
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, or PNG files are supported.")
    
    try:
        # Membaca file gambar
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)

        # Deteksi wajah menggunakan MTCNN
        boxes, _ = detector.detect(image)
        if boxes is None or len(boxes) == 0:
            raise HTTPException(status_code=400, detail="No faces detected.")

        # Loop untuk proses klasifikasi dan anotasi gambar
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image_np[y1:y2, x1:x2]
            face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            # Prediksi ekspresi
            predictions = model.predict(face_reshaped)
            emotion = emotion_labels[np.argmax(predictions)]

            # Tambahkan bounding box dan label pada gambar
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image_np, emotion, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        # Encode gambar menjadi JPEG
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

@app.post("/get-expression-label/")
@limiter.limit("10/minute")
async def get_expression_label(request: Request, file: UploadFile = File(...)):
    """
    Endpoint untuk mendeteksi ekspresi wajah dan hanya mengembalikan label ekspresi.
    """
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG, JPEG, or PNG files are supported.")
    
    try:
        # Membaca file gambar
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)

        # Deteksi wajah menggunakan MTCNN
        boxes, _ = detector.detect(image)
        if boxes is None or len(boxes) == 0:
            raise HTTPException(status_code=400, detail="No faces detected.")

        expression_results = []

        # Loop untuk proses klasifikasi
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = image_np[y1:y2, x1:x2]
            face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            # Prediksi ekspresi
            predictions = model.predict(face_reshaped)
            emotion = emotion_labels[np.argmax(predictions)]
            expression_results.append(emotion)

        return JSONResponse({"emotions": expression_results}, status_code=200)

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
