from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN
import cv2
from io import BytesIO

# Inisialisasi aplikasi dan limiter
app = FastAPI(title="Face Expression Detection API")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Tambahkan middleware
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
    	"modelekspresi-production.up.railway.app",
    	"localhost",
    	"localhost:8000"
	]
)

# Handler untuk rate limit
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    response = PlainTextResponse(
        "Rate limit exceeded. Please try again in 60 seconds.",
        status_code=429
    )
    response.headers["Retry-After"] = "60"
    return response

# Model TensorFlow
model = None
emotion_labels = ["Angry", "Fear", "Happy", "Neutral", "Sad"]

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model("model/expression_detection_new.h5")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Inisialisasi MTCNN
detector = MTCNN(keep_all=True)

@app.get("/")
async def read_root():
    """
    Root endpoint untuk health check
    """
    return {"status": "active", "message": "Face Expression Detection API is running"}

@app.post("/detect-expression/")
@limiter.limit("10/minute")
async def detect_expression(request: Request, file: UploadFile = File(...)):
    """
    Endpoint untuk mendeteksi ekspresi wajah dari gambar.
    Returns gambar dengan anotasi bounding box dan label ekspresi.
    """
    # Validasi file
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, or PNG files are supported."
        )
    
    try:
        # Membaca file gambar
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)

        # Deteksi wajah menggunakan MTCNN
        boxes, _ = detector.detect(image)
        if boxes is None or len(boxes) == 0:
            raise HTTPException(
                status_code=400,
                detail="No faces detected in the image."
            )

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
            confidence = float(np.max(predictions)) * 100

            # Tambahkan bounding box dan label pada gambar
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{emotion} ({confidence:.1f}%)"
            cv2.putText(
                image_np,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        # Encode gambar menjadi JPEG
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/jpeg"
        )

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the image: {str(e)}"
        )

@app.post("/get-expression-label/")
@limiter.limit("10/minute")
async def get_expression_label(request: Request, file: UploadFile = File(...)):
    """
    Endpoint untuk mendeteksi ekspresi wajah dan mengembalikan label ekspresi.
    Returns JSON dengan daftar ekspresi yang terdeteksi.
    """
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, or PNG files are supported."
        )
    
    try:
        # Membaca file gambar
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(image, dtype=np.uint8)

        # Deteksi wajah menggunakan MTCNN
        boxes, _ = detector.detect(image)
        if boxes is None or len(boxes) == 0:
            raise HTTPException(
                status_code=400,
                detail="No faces detected in the image."
            )

        expression_results = []

        # Loop untuk proses klasifikasi
        for i, box in enumerate(boxes, 1):
            x1, y1, x2, y2 = map(int, box)
            face = image_np[y1:y2, x1:x2]
            face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

            # Prediksi ekspresi
            predictions = model.predict(face_reshaped)
            emotion = emotion_labels[np.argmax(predictions)]
            confidence = float(np.max(predictions)) * 100
            
            expression_results.append({
                "face_id": i,
                "emotion": emotion,
                "confidence": f"{confidence:.1f}%"
            })

        return JSONResponse({
            "status": "success",
            "faces_detected": len(expression_results),
            "results": expression_results
        })

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the image: {str(e)}"
        )

# Error Handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected error occurred",
            "detail": str(exc)
        }
    )
