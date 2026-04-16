import os
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI(title="Malaria Cell Classification API")

# CORS (allow Streamlit to talk to API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Paths
MODELS_DIR = "models"
UPLOAD_DIR = "data/uploads"
CLASSES = ["Parasitized", "Uninfected"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global model
model = None


# ✅ Load Random Forest model
def load_prediction_model():
    global model
    model_path = os.path.join(MODELS_DIR, "rf_model.pkl")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Random Forest model loaded.")
    else:
        print("No model found.")


load_prediction_model()


# Root
@app.get("/")
def root():
    return {"message": "Malaria API running"}


# Health check
@app.get("/health")
def health():
    return {
        "status": "online",
        "model": "loaded" if model is not None else "not loaded"
    }


# ✅ Prediction endpoint (FIXED)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((64, 64))

        img_array = np.array(image) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)

        if model is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded"})

        pred = model.predict(img_flat)[0]
        prob = model.predict_proba(img_flat)[0][1]

        label = CLASSES[pred]
        confidence = prob if pred == 1 else (1 - prob)

        return {
            "filename": file.filename,
            "label": label,
            "confidence": round(float(confidence) * 100, 2)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Upload images (unchanged)
@app.post("/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    label: str = "Parasitized"
):
    if label not in CLASSES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Label must be one of {CLASSES}"}
        )

    saved = []
    for file in files:
        path = os.path.join(UPLOAD_DIR, label, file.filename)
        with open(path, "wb") as f:
            f.write(await file.read())
        saved.append(file.filename)

    return {
        "message": f"{len(saved)} images uploaded.",
        "files": saved
    }


# Metrics
@app.get("/metrics")
def metrics():
    return {
        "model": "Random Forest",
        "status": "loaded" if model is not None else "not loaded"
    }