import os
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="Malaria Cell Classification API")

# CORS (allow Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODELS_DIR = "models"
CLASSES = ["Parasitized", "Uninfected"]

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Global model
model = None


# ✅ Load model (Random Forest)
def load_prediction_model():
    global model
    model_path = os.path.join(MODELS_DIR, "rf_model.pkl")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
    else:
        print("❌ Model file not found:", model_path)


load_prediction_model()


# Root endpoint
@app.get("/")
def root():
    return {"message": "Malaria API running"}


# Health check
@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "loaded" if model is not None else "not loaded",
    }


# ✅ Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Load and preprocess image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((64, 64))

        img_array = np.array(image) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)

        # Check model
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Model not loaded"}
            )

        # Predict
        pred = model.predict(img_flat)[0]
        prob = model.predict_proba(img_flat)[0][1]

        label = CLASSES[pred]
        confidence = prob if pred == 1 else (1 - prob)

        return {
            "filename": file.filename,
            "label": label,
            "confidence": round(float(confidence) * 100, 2),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# Metrics endpoint
@app.get("/metrics")
def metrics():
    return {
        "model": "Random Forest",
        "status": "loaded" if model is not None else "not loaded",
    }