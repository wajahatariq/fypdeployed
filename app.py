import streamlit as st
from pathlib import Path
import os
import cv2
import numpy as np
import easyocr
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import random
import string
from ultralytics import YOLO

# -------------------- Paths & Config --------------------
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
MODEL_PATH = BASE_DIR / "license_plate_detector.pt"  # your new model

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# -------------------- Load OCR & Model --------------------
reader = easyocr.Reader(['en'], gpu=False)

@st.cache_resource(ttl=3600)
def load_yolo_model():
    return YOLO(MODEL_PATH)

model = load_yolo_model()

# -------------------- Helper Functions --------------------
def preprocess_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_number_plate(image_path: Path) -> str | None:
    processed = preprocess_image(image_path)
    results = reader.readtext(processed)
    texts = [text for _, text, conf in results if conf > 0.4]
    return " ".join(texts).strip() if texts else None

def detect_number_plate(image_path: Path) -> list[str]:
    results = model.predict(source=str(image_path), save=False, conf=0.3, verbose=False)
    detected = []
    if results and len(results) > 0:
        for r in results:
            for cls in r.boxes.cls.cpu().numpy():
                detected.append(model.names[int(cls)])
    return detected

def generate_challan_pdf(plate_text: str, challan_id: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Traffic Violation E-Challan")

    c.setFont("Helvetica", 14)
    c.drawString(50, height - 100, f"Challan ID: {challan_id}")
    c.drawString(50, height - 130, f"Vehicle Number: {plate_text or 'UNKNOWN'}")
    c.drawString(50, height - 160, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def generate_challan_id(length=8) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# -------------------- Streamlit App --------------------
def main():
    st.set_page_config(page_title="Number Plate Detection", layout="centered")

    # Load custom CSS
    css_path = BASE_DIR / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title("🚦 Number Plate Detection & E-Challan Generator")

    uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_path = TEMP_DIR / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(str(temp_path), caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting number plate..."):
            plate_text = extract_number_plate(temp_path)
        st.markdown(f"**Extracted Number Plate:** {plate_text or 'Not detected'}")

        with st.spinner("Detecting number plate bounding box..."):
            detected = detect_number_plate(temp_path)
        if detected:
            st.markdown("**Number Plate Detected:**")
            for d in detected:
                st.write(f"- {d}")
        else:
            st.info("No number plate detected.")

        # Generate PDF Challan if detected
        if plate_text or detected:
            challan_id = generate_challan_id()
            pdf_buffer = generate_challan_pdf(plate_text or "UNKNOWN", challan_id)
            st.download_button(
                label="Download Challan PDF",
                data=pdf_buffer,
                file_name=f"challan_{challan_id}.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()

