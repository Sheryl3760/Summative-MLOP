import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.preprocessing import preprocess_single_image

CLASSES = ["Parasitized", "Uninfected"]


def load_best_model(models_dir):
    model = load_model(f"{models_dir}/vgg16_model.h5")
    return model


def predict_image(image_path, models_dir="models"):
    img_array, img_flat, img_dl = preprocess_single_image(image_path)

    model = load_best_model(models_dir)
    prob  = model.predict(img_dl)[0][0]

    label = CLASSES[0] if prob < 0.5 else CLASSES[1]
    confidence = (1 - prob) if prob < 0.5 else prob

    return {
        "label"     : label,
        "confidence": round(float(confidence) * 100, 2),
        "raw_prob"  : round(float(prob), 4)
    }