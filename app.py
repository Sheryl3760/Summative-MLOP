import streamlit as st
import requests
from PIL import Image

API_URL = "https://summative-mlop-1-gwpb.onrender.com"

st.set_page_config(
    page_title="Malaria Cell Classification",
    layout="wide"
)

st.title("Malaria Cell Classification Dashboard")
st.markdown("---")


def check_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=10)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return None


def get_metrics():
    try:
        res = requests.get(f"{API_URL}/metrics", timeout=10)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return None


health = check_health()
metrics = get_metrics()

api_online = health is not None and health.get("status") == "online"

col1, col2, col3, col4 = st.columns(4)

with col1:
    if api_online:
        st.success("API Status: Online")
    else:
        st.error("API Status: Offline")

with col2:
    model_status = health.get("model") if health else "unknown"
    st.info(f"Model: {model_status}")

with col3:
    if metrics:
        st.info(f"Model Name: {metrics.get('model', 'Unknown')}")
    else:
        st.info("Model Name: Unknown")

with col4:
    if st.button("Refresh"):
        st.rerun()

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Predict",
    "Visualizations",
    "Upload Data",
    "Retrain Model"
])

with tab1:
    st.header("Predict a Cell Image")

    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(Image.open(file), width=250)

        with col2:
            if st.button("Predict"):
                try:
                    res = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (file.name, file.getvalue(), "image/png")},
                        timeout=20
                    )

                    data = res.json()

                    if "error" in data:
                        st.error(data["error"])
                    else:
                        label = data.get("label", "Unknown")
                        confidence = data.get("confidence", 0)

                        if label == "Parasitized":
                            st.error(f"Prediction: {label}")
                        else:
                            st.success(f"Prediction: {label}")

                        st.metric("Confidence", f"{confidence}%")

                except:
                    st.error("Could not connect to API")

with tab2:
    st.header("Dataset Overview")
    st.write("Balanced dataset with Parasitized and Uninfected cells.")

with tab3:
    st.header("Upload Data")

    label = st.selectbox("Select label", ["Parasitized", "Uninfected"])

    files = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if files:
        st.write(f"{len(files)} files selected")

        if st.button("Upload"):
            try:
                payload = [
                    ("files", (f.name, f.getvalue(), "image/png"))
                    for f in files
                ]

                res = requests.post(
                    f"{API_URL}/upload",
                    files=payload,
                    data={"label": label},
                    timeout=30
                )

                st.success(res.json().get("message", "Upload complete"))

            except:
                st.error("Upload failed")

with tab4:
    st.header("Retrain Model")

    if st.button("Start Retraining"):
        try:
            res = requests.post(f"{API_URL}/retrain", timeout=10)
            st.success(res.json().get("message", "Retraining started"))
        except:
            st.error("Could not start retraining")