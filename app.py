import streamlit as st
import requests
import os
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Malaria Cell Classification",
    layout="wide"
)

st.markdown("""
    <style>
        .main { background-color: #f0f2f5; }
        .block-container { padding-top: 2rem; }
        h1 { color: #1a1a2e; }
        h2 { color: #16213e; }
        .stButton>button {
            background-color: #1a1a2e;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            width: 100%;
        }
        .stButton>button:hover { background-color: #16213e; }
    </style>
""", unsafe_allow_html=True)

st.title("Malaria Cell Classification — Monitoring Dashboard")
st.markdown("---")

def check_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        return res.json()
    except:
        return None

def get_metrics():
    try:
        res = requests.get(f"{API_URL}/metrics", timeout=3)
        return res.json()
    except:
        return None

health = check_health()
metrics = get_metrics()

col1, col2, col3, col4 = st.columns(4)

with col1:
    if health:
        st.success("API Status: Online")
    else:
        st.error("API Status: Offline")

with col2:
    if health:
        model_status = health.get("model", "unknown")
        if model_status == "loaded":
            st.success(f"Model: {model_status}")
        else:
            st.warning(f"Model: {model_status}")
    else:
        st.error("Model: Unknown")

with col3:
    if metrics:
        st.info(f"Model Name: {metrics.get('model', 'Unknown')}")
    else:
        st.info("Model Name: Unknown")

with col4:
    if st.button("Refresh Status"):
        st.rerun()

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Predict",
    "Visualizations",
    "Upload Data",
    "Retrain Model"
])

with tab1:
    st.header("Predict a Single Cell Image")
    st.write("Upload a microscopy image of a blood cell to classify it as Parasitized or Uninfected.")

    uploaded_file = st.file_uploader(
        "Choose a cell image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, width=250)

        with col2:
            st.subheader("Prediction Result")
            if st.button("Run Prediction"):
                with st.spinner("Running prediction..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/png")}
                        res   = requests.post(f"{API_URL}/predict", files=files)
                        data  = res.json()

                        if "error" in data:
                            st.error(f"Error: {data['error']}")
                        else:
                            label      = data.get("label", "Unknown")
                            confidence = data.get("confidence", 0)

                            if label == "Parasitized":
                                st.error(f"Prediction: {label}")
                            else:
                                st.success(f"Prediction: {label}")

                            st.metric("Confidence", f"{confidence}%")
                            st.metric("Raw Probability", data.get("raw_prob", 0))

                    except Exception as e:
                        st.error(f"Could not connect to API: {e}")

with tab2:
    st.header("Dataset Visualizations")
    st.write("Visual analysis of the NIH Malaria Cell Images dataset.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        classes = ["Parasitized", "Uninfected"]
        counts  = [13779, 13779]
        colors  = ["#d9534f", "#5cb85c"]
        ax.bar(classes, counts, color=colors, edgecolor="black", alpha=0.85)
        ax.set_ylabel("Number of Images")
        ax.set_title("Image Count per Class")
        for i, count in enumerate(counts):
            ax.text(i, count + 100, str(count), ha="center", fontweight="bold")
        st.pyplot(fig)
        st.caption("The dataset is perfectly balanced with 13,779 images per class.")

    with col2:
        st.subheader("Model Performance Comparison")
        model_data = {
            "Model": ["Sequential CNN", "Functional CNN", "VGG16 Transfer", "Random Forest", "SVM"],
            "F1 Score": [0.9598, 0.9549, 0.9354, 0.8073, 0.7114],
            "Accuracy": [0.9590, 0.9539, 0.9343, 0.8135, 0.7074],
            "AUC-ROC":  [0.9895, 0.9866, 0.9833, 0.8939, 0.7766]
        }
        df = pd.DataFrame(model_data)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(df["Model"], df["F1 Score"], color="#2196F3", alpha=0.85, label="F1 Score")
        ax.set_xlabel("Score")
        ax.set_title("Model F1 Score Comparison")
        ax.set_xlim(0, 1)
        st.pyplot(fig)
        st.caption("Deep learning models significantly outperform traditional ML on image data.")

    st.subheader("Detailed Model Metrics Table")
    df_display = pd.DataFrame(model_data)
    st.dataframe(df_display, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Accuracy vs F1 Score")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            df["Accuracy"], df["F1 Score"],
            c=["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"],
            s=200, zorder=5
        )
        for i, row in df.iterrows():
            ax.annotate(
                row["Model"],
                (row["Accuracy"], row["F1 Score"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8
            )
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("F1 Score")
        ax.set_title("Accuracy vs F1 Score per Model")
        ax.plot([0.6, 1.0], [0.6, 1.0], "k--", alpha=0.3)
        st.pyplot(fig)
        st.caption("Models closer to the top right corner have both high accuracy and high F1 score.")

    with col4:
        st.subheader("AUC-ROC per Model")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
        bars    = ax.bar(df["Model"], df["AUC-ROC"], color=colors, edgecolor="black", alpha=0.85)
        ax.set_ylabel("AUC-ROC Score")
        ax.set_title("AUC-ROC per Model")
        ax.set_ylim(0.6, 1.0)
        ax.set_xticklabels(df["Model"], rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, df["AUC-ROC"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center", fontsize=8
            )
        st.pyplot(fig)
        st.caption("AUC-ROC above 0.98 indicates excellent discriminative ability.")

with tab3:
    st.header("Upload Training Data")
    st.write("Upload new cell images to be used for retraining the model.")

    label = st.selectbox(
        "Select the class label for these images",
        ["Parasitized", "Uninfected"]
    )

    uploaded_files = st.file_uploader(
        "Choose images to upload",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} image(s) selected for class: {label}")

        if st.button("Upload Images"):
            with st.spinner("Uploading..."):
                try:
                    files = [
                        ("files", (f.name, f.getvalue(), "image/png"))
                        for f in uploaded_files
                    ]
                    res  = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        data={"label": label}
                    )
                    data = res.json()
                    st.success(data.get("message", "Upload complete."))
                    st.write("Uploaded files:", data.get("files", []))
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")

with tab4:
    st.header("Retrain the Model")
    st.write("""
        Upload new images using the Upload Data tab first, then trigger retraining here.
        Retraining runs in the background and may take several minutes depending on
        the amount of new data uploaded.
    """)

    st.warning("Make sure you have uploaded new training data before triggering retraining.")

    if st.button("Trigger Retraining"):
        with st.spinner("Sending retraining request..."):
            try:
                res  = requests.post(f"{API_URL}/retrain")
                data = res.json()
                st.success(data.get("message", "Retraining triggered."))
                st.info("The model is retraining in the background. Check back in a few minutes.")
            except Exception as e:
                st.error(f"Could not connect to API: {e}")