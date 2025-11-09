import streamlit as st
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

# Config and paths
OUTPUT_DIR = "output"
TEMP_DIR = "temp"
MODEL_PATH = "best.pt"  # Assuming best.pt is in the root of your repo

# Ensure folders exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Load EasyOCR reader once (English)
reader = easyocr.Reader(['en'], gpu=False)

# Load YOLO model once, cached for performance
@st.cache_resource(ttl=3600)
def load_yolo_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_yolo_model()

def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_number_plate(image_path: str) -> str | None:
    processed = preprocess_image(image_path)
    results = reader.readtext(processed)
    texts = [text for _, text, conf in results if conf > 0.4]
    return " ".join(texts).strip() if texts else None

def analyze_violations(image_path: str) -> list[str]:
    results = model.predict(source=image_path, save=False, conf=0.3, verbose=False)
    detected_classes = []
    if results and len(results) > 0:
        for r in results:
            for cls in r.boxes.cls.cpu().numpy():
                class_name = model.names[int(cls)]
                detected_classes.append(class_name)
    return detected_classes

def generate_challan_pdf(vehicle_number: str, violations: list[str], fine_amount: int, challan_id: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Traffic Violation E-Challan")

    c.setFont("Helvetica", 14)
    c.drawString(50, height - 100, f"Challan ID: {challan_id}")
    c.drawString(50, height - 130, f"Vehicle Number: {vehicle_number}")
    c.drawString(50, height - 160, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.drawString(50, height - 200, "Violations:")
    y = height - 230
    for v in violations:
        c.drawString(70, y, f"- {v}")
        y -= 20

    c.drawString(50, y - 20, f"Total Fine Amount: ₹{fine_amount}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def generate_challan_id(length=8) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# Adjust this dictionary to your exact class names from the model
FINE_AMOUNTS = {
    "without helmet": 500,
    "no seatbelt": 300,
    "triple riding": 700,
    "number plate": 0,  # usually no fine for number plate detection itself
    # add more if needed
}

from pathlib import Path

def main():
    st.set_page_config(page_title="Traffic Violation Detection & E-Challan", layout="centered")

    # Load your custom CSS from style.css using pathlib
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        css_content = css_path.read_text()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    st.title("Traffic Violation Detection & E-Challan Generator")

    # ... rest of your code unchanged ...


    uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(temp_path, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting number plate..."):
            plate = extract_number_plate(temp_path)
        st.markdown(f"**Extracted Number Plate:** {plate or 'Not detected'}")

        with st.spinner("Analyzing for violations..."):
            violations = analyze_violations(temp_path)

        if violations:
            st.markdown("**Detected Violations:**")
            for v in violations:
                st.write(f"- {v}")
        else:
            st.info("No violations detected.")

        total_fine = sum(FINE_AMOUNTS.get(v.lower(), 0) for v in violations)
        st.markdown(f"**Total Fine Amount:** ₹{total_fine}")

        if violations:
            challan_id = generate_challan_id()
            pdf_buffer = generate_challan_pdf(plate or "UNKNOWN", violations, total_fine, challan_id)

            st.download_button(
                label="Download Challan PDF",
                data=pdf_buffer,
                file_name=f"challan_{challan_id}.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
