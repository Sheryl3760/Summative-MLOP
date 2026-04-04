import os
import shutil
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import io
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_single_image, load_dataset, flatten_images
from src.model import (
    build_vgg16_model,
    train_deep_learning_model,
    train_random_forest,
    train_svm,
    save_models
)

app = FastAPI(title="Malaria Cell Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODELS_DIR   = "models"
UPLOAD_DIR   = "data/uploads"
CLASSES      = ["Parasitized", "Uninfected"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/Parasitized", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/Uninfected", exist_ok=True)

model = None


def load_prediction_model():
    global model
    model_path = os.path.join(MODELS_DIR, "vgg16_model.h5")
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Model loaded successfully.")
    else:
        print("No model found. Train the model first.")


load_prediction_model()


@app.get("/")
def root():
    return {"message": "Malaria Cell Classification API is running."}


@app.get("/health")
def health():
    return {
        "status" : "online",
        "model"  : "loaded" if model is not None else "not loaded"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents  = await file.read()
        image     = Image.open(io.BytesIO(contents)).convert("RGB")
        image     = image.resize((64, 64))
        img_array = np.array(image) / 255.0
        img_dl    = np.expand_dims(img_array, axis=0)

        if model is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded."})

        prob      = model.predict(img_dl)[0][0]
        label     = CLASSES[0] if prob < 0.5 else CLASSES[1]
        confidence = (1 - prob) if prob < 0.5 else prob

        return {
            "filename"  : file.filename,
            "label"     : label,
            "confidence": round(float(confidence) * 100, 2),
            "raw_prob"  : round(float(prob), 4)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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
        dest = os.path.join(UPLOAD_DIR, label, file.filename)
        with open(dest, "wb") as f:
            f.write(await file.read())
        saved.append(file.filename)

    return {
        "message": f"{len(saved)} images uploaded to {label} folder.",
        "files"  : saved
    }


def retrain_task():
    global model

    print("Retraining started...")
    images, labels = load_dataset(UPLOAD_DIR)

    if len(images) == 0:
        print("No images found for retraining.")
        return

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    new_model = build_vgg16_model((64, 64, 3))
    train_deep_learning_model(new_model, X_train, y_train)
    new_model.save(os.path.join(MODELS_DIR, "vgg16_model.h5"))

    X_flat   = flatten_images(X_train)
    rf_model = train_random_forest(X_flat, y_train)
    svm_model = train_svm(X_flat, y_train)

    joblib.dump(rf_model,  os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(svm_model, os.path.join(MODELS_DIR, "svm_model.pkl"))

    model = new_model
    print("Retraining complete.")


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_task)
    return {"message": "Retraining started in the background."}


@app.get("/metrics")
def metrics():
    return {
        "model"      : "VGG16 Transfer Learning",
        "status"     : "loaded" if model is not None else "not loaded",
        "upload_dir" : UPLOAD_DIR,
        "models_dir" : MODELS_DIR
    }