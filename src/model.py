import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

IMG_SIZE    = (64, 64)
BATCH_SIZE  = 32
RANDOM_SEED = 42
CLASSES     = ["Parasitized", "Uninfected"]


def build_sequential_cnn(input_shape=(64, 64, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name="Sequential_CNN")

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_functional_cnn(input_shape=(64, 64, 3)):
    inputs = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="Functional_CNN")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_vgg16_model(input_shape=(64, 64, 3)):
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs  = Input(shape=input_shape)
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs, name="VGG16_Transfer")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]


def train_deep_learning_model(model, X_train, y_train):
    y_train_dl = y_train.astype(np.float32)
    history = model.fit(
        X_train, y_train_dl,
        validation_split=0.2,
        epochs=30,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1
    )
    return history


def train_random_forest(X_train_flat, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_model.fit(X_train_flat, y_train)
    return rf_model


def train_svm(X_train_flat, y_train, sample_size=5000):
    X_sample, _, y_sample, _ = train_test_split(
        X_train_flat, y_train,
        train_size=sample_size,
        random_state=RANDOM_SEED,
        stratify=y_train
    )
    svm_model = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
    svm_model.fit(X_sample, y_sample)
    return svm_model


def save_models(models_dir, rf_model, svm_model, sequential_cnn, functional_cnn, vgg16_model):
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(svm_model, os.path.join(models_dir, "svm_model.pkl"))
    sequential_cnn.save(os.path.join(models_dir, "sequential_cnn.h5"))
    functional_cnn.save(os.path.join(models_dir, "functional_cnn.h5"))
    vgg16_model.save(os.path.join(models_dir, "vgg16_model.h5"))

    print("All models saved to", models_dir)


def load_models(models_dir):
    from tensorflow.keras.models import load_model

    rf_model       = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    svm_model      = joblib.load(os.path.join(models_dir, "svm_model.pkl"))
    sequential_cnn = load_model(os.path.join(models_dir, "sequential_cnn.h5"))
    functional_cnn = load_model(os.path.join(models_dir, "functional_cnn.h5"))
    vgg16_model    = load_model(os.path.join(models_dir, "vgg16_model.h5"))

    print("All models loaded from", models_dir)
    return rf_model, svm_model, sequential_cnn, functional_cnn, vgg16_model