import os
import numpy as np
from PIL import Image

IMG_SIZE    = (64, 64)
CLASSES     = ["Parasitized", "Uninfected"]


def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return img_array


def load_dataset(data_dir):
    images = []
    labels = []

    for label, cls in enumerate(CLASSES):
        class_path = os.path.join(data_dir, cls)

        if not os.path.exists(class_path):
            print(f"Warning: {class_path} does not exist. Skipping.")
            continue

        files = [
            f for f in os.listdir(class_path)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"Loading {cls}: {len(files)} images...")

        for fname in files:
            img_path = os.path.join(class_path, fname)
            try:
                img_array = load_and_preprocess_image(img_path)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Skipped {fname}: {e}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"Dataset loaded: {images.shape[0]} images total.")
    return images, labels


def flatten_images(images):
    return images.reshape(images.shape[0], -1)


def preprocess_single_image(image_path):
    img_array = load_and_preprocess_image(image_path)
    img_flat  = img_array.flatten().reshape(1, -1)
    img_dl    = np.expand_dims(img_array, axis=0)
    return img_array, img_flat, img_dl