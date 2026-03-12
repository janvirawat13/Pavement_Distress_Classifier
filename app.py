import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("Project Info")
st.sidebar.info(
"""
AI system for detecting **road damages** such as:

• Potholes  
• Cracks  
• Surface Distress  

Upload an image and the model will analyze it.
"""
)

# ---------------------------
# Title
# ---------------------------
st.title("🛣️ AI Road Damage Detection System")
st.write("Upload a road image and detect potholes or road damage using YOLO.")

# ---------------------------
# Load YOLO Model
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = YOLO(r"D:\image_classifiction\runs\detect\train2\weights\best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "png", "jpeg"]
)

# ---------------------------
# Process Image
# ---------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    img_array = np.array(image)

    if st.button("Detect Road Damage"):

        if model is None:
            st.error("Model not loaded.")
        else:

            with st.spinner("Running AI Detection..."):

                results = model(img_array)

                annotated_frame = results[0].plot()

                with col2:
                    st.subheader("Detection Result")
                    st.image(annotated_frame, use_container_width=True)

                boxes = results[0].boxes

                # ---------------------------
                # Detection Details
                # ---------------------------
                if boxes is not None and len(boxes) > 0:

                    st.subheader("Detection Details")

                    detection_count = 0

                    for box in boxes:

                        conf = float(box.conf)
                        cls = int(box.cls)
                        label = model.names[cls]

                        detection_count += 1

                        st.write(
                            f"**Damage Type:** {label} | **Confidence:** {conf:.2f}"
                        )

                    st.success(f"Total Damages Detected: {detection_count}")

                else:
                    st.warning("No road damage detected.")

                # ---------------------------
                # Download Result Image
                # ---------------------------
                result_image = Image.fromarray(annotated_frame)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

                    result_image.save(tmp.name)

                    with open(tmp.name, "rb") as file:

                        st.download_button(
                            label="Download Result Image",
                            data=file,
                            file_name="detected_road_damage.png",
                            mime="image/png"
                        )