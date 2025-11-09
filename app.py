import streamlit as st
import os
import cv2
import numpy as np
import easyocr
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from datetime import datetime
import random
import string
from ultralytics import YOLO
import qrcode
from PIL import Image
from pathlib import Path

OUTPUT_DIR = "output"
TEMP_DIR = "temp"
MODEL_PATH = "best.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False)

@st.cache_resource(ttl=3600)
def load_yolo_model():
    return YOLO(MODEL_PATH)

model = load_yolo_model()

def preprocess_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    enhanced = cv2.equalizeHist(sharpened)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_number_plate(image_path: str, conf_threshold=0.3) -> tuple[str | None, np.ndarray | None]:
    processed = preprocess_image(image_path)
    results = reader.readtext(processed)
    texts = []
    img_draw = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    for bbox, text, conf in results:
        if conf > conf_threshold:
            texts.append(text)
            pts = np.array(bbox).astype(int)
            cv2.polylines(img_draw, [pts], True, (0,255,0), 2)
            cv2.putText(img_draw, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    extracted_text = " ".join(texts).strip() if texts else None
    return extracted_text, img_draw

def analyze_violations(image_path: str) -> list[str]:
    results = model.predict(source=image_path, save=False, conf=0.3, verbose=False)
    detected_classes = []
    if results and len(results) > 0:
        for r in results:
            for cls in r.boxes.cls.cpu().numpy():
                class_name = model.names[int(cls)]
                detected_classes.append(class_name)
    return detected_classes

def generate_qr_code(data: str) -> Image.Image:
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    return img

def generate_challan_pdf(vehicle_number: str, violations: list[str], fine_amount: int, challan_id: str, vehicle_image_path: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Define colors
    primary_color = colors.HexColor("#004085")   # Navy Blue
    secondary_color = colors.HexColor("#d6e0f0") # Light Blue Background
    alert_color = colors.HexColor("#dc3545")     # Red for important
    text_color = colors.black

    margin = 40
    y = height - margin

    # Background block header
    c.setFillColor(secondary_color)
    c.rect(0, y - 80, width, 80, fill=True, stroke=False)

    # Header Title
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(width / 2, y - 50, "Traffic Violation E-Challan")

    y -= 110

    # Info box background
    c.setFillColor(secondary_color)
    c.roundRect(margin, y - 110, width - 2 * margin, 110, 10, fill=True, stroke=False)

    # Info Text
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin + 10, y - 40, f"Challan ID: ")
    c.setFont("Helvetica", 14)
    c.drawString(margin + 110, y - 40, challan_id)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin + 10, y - 65, "Vehicle Number: ")
    c.setFont("Helvetica", 14)
    c.drawString(margin + 140, y - 65, vehicle_number)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin + 10, y - 90, "Date: ")
    c.setFont("Helvetica", 14)
    c.drawString(margin + 60, y - 90, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    y -= 150

    # Vehicle Image
    if os.path.exists(vehicle_image_path):
        max_width = 220
        max_height = 160
        c.setFillColor(text_color)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y + 20, "Vehicle Image:")
        c.drawImage(ImageReader(vehicle_image_path), margin, y - max_height + 10, width=max_width, height=max_height, preserveAspectRatio=True)

    # Violations box background
    viol_x = margin + 250
    viol_y = y + 120
    viol_w = width - viol_x - margin
    viol_h = max_height + 20

    c.setFillColor(secondary_color)
    c.roundRect(viol_x, y - 10, viol_w, viol_h, 10, fill=True, stroke=False)

    # Violations Title
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(viol_x + 10, viol_y, "Violations and Fines:")

    c.setFont("Helvetica", 14)
    line_y = viol_y - 30
    for v in violations:
        fine = VIOLATION_DETAILS.get(v.lower(), {}).get("fine", 0)
        icon = VIOLATION_DETAILS.get(v.lower(), {}).get("icon", "")
        # Red color for serious violations
        c.setFillColor(alert_color if VIOLATION_DETAILS.get(v.lower(), {}).get("severity") == "serious" else text_color)
        c.drawString(viol_x + 20, line_y, f"{icon} {v}: ₹{fine}")
        line_y -= 25

    # Total fine
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(viol_x + 20, line_y - 10, f"Total Fine Amount: ₹{fine_amount}")

    # QR Code bottom right
    qr_img = generate_qr_code(challan_id)
    qr_buffer = BytesIO()
    qr_img.save(qr_buffer)
    qr_buffer.seek(0)

    qr_size = 100
    c.drawImage(ImageReader(qr_buffer), width - margin - qr_size, margin, width=qr_size, height=qr_size)

    # Footer Text
    c.setFillColor(colors.grey)
    c.setFont("Helvetica-Oblique", 10)
    c.drawCentredString(width / 2, margin / 2, "Generated by Traffic Violation System")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def generate_challan_id(length=8) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

VIOLATION_DETAILS = {
    "without helmet": {
        "fine": 500,
        "icon": "🪖",
        "severity": "serious",
        "desc": "Riding without a helmet puts your safety at risk and is punishable by fine."
    },
    "no seatbelt": {
        "fine": 300,
        "icon": "🔒",
        "severity": "warning",
        "desc": "Seatbelt ensures safety in accidents; non-usage attracts fines."
    },
    "triple riding": {
        "fine": 700,
        "icon": "👥",
        "severity": "serious",
        "desc": "Carrying more than two passengers is illegal and dangerous."
    },
    "number plate": {
        "fine": 0,
        "icon": "🔢",
        "severity": "info",
        "desc": "Number plate detection only."
    }
}

def main():
    st.set_page_config(page_title="Traffic Violation Detection & E-Challan", layout="centered")

    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        css_content = css_path.read_text()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    st.title("Traffic Violation Detection & E-Challan Generator")

    uploaded_file = st.file_uploader("Upload Vehicle Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(temp_path, caption="Uploaded Vehicle Image", use_column_width=True)

        with col2:
            with st.spinner("Extracting number plate and OCR bounding boxes..."):
                plate, ocr_img = extract_number_plate(temp_path, conf_threshold=0.3)
            if ocr_img is not None:
                st.image(ocr_img, caption="OCR Bounding Boxes on Preprocessed Image", use_column_width=True)
            st.markdown(f"**Extracted Number Plate:** {plate or 'Not detected'}")

        with st.spinner("Analyzing for violations..."):
            violations = analyze_violations(temp_path)

        if violations:
            st.markdown("**Detected Violations:**")
            for v in violations:
                details = VIOLATION_DETAILS.get(v.lower(), {"fine":0, "icon":"❓", "severity":"info", "desc":""})
                badge_class = details["severity"]
                icon = details["icon"]
                desc = details["desc"]
                fine = details["fine"]

                st.markdown(
                    f"""
                    <div class="violation-badge {badge_class}" title="{desc}">
                        <span class="violation-icon">{icon}</span> {v} - ₹{fine}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No violations detected.")

        total_fine = sum(VIOLATION_DETAILS.get(v.lower(), {}).get("fine", 0) for v in violations)
        st.markdown(f"### Total Fine Amount: ₹{total_fine}")

        if violations:
            challan_id = generate_challan_id()
            pdf_buffer = generate_challan_pdf(plate or "UNKNOWN", violations, total_fine, challan_id, temp_path)

            st.download_button(
                label="Download Challan PDF",
                data=pdf_buffer,
                file_name=f"challan_{challan_id}.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()
