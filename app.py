import streamlit as st
import os
import cv2
import numpy as np
import easyocr
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.enums import TA_CENTER
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
    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    return img

def generate_challan_pdf(vehicle_number: str, violations: list[str], fine_amount: int, challan_id: str, vehicle_image_path: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    primary_color = colors.HexColor("#0B3D91")  # Deep Blue
    secondary_color = colors.HexColor("#F24C00")  # Orange/red
    styles = getSampleStyleSheet()
    styleN = styles["Normal"]
    styleB = styles["Heading2"]
    styleB.alignment = TA_CENTER

    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 28)
    c.drawCentredString(width / 2, height - 60, "Traffic Violation E-Challan")

    c.setStrokeColor(primary_color)
    c.setLineWidth(2)
    c.line(40, height - 70, width - 40, height - 70)

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Challan ID: {challan_id}")
    c.drawString(50, height - 120, f"Vehicle Number: {vehicle_number}")
    c.drawString(50, height - 140, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if os.path.exists(vehicle_image_path):
        max_width = 240
        max_height = 180
        c.drawString(50, height - 170, "Vehicle Image:")
        c.drawImage(ImageReader(vehicle_image_path), 50, height - 350, width=max_width, height=max_height, preserveAspectRatio=True)

    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(320, height - 170, "Violations and Fines")

    table_data = [["Violation", "Description", "Fine (₹)"]]
    for v in violations:
        details = VIOLATION_DETAILS.get(v.lower(), {"fine":0, "desc":"", "icon":"❓"})
        table_data.append([v, details["desc"], f"₹{details['fine']}"])

    table = Table(table_data, colWidths=[100, 230, 80])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), primary_color),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 14),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BOX", (0,0), (-1,-1), 0.75, colors.black),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,1), (-1,-1), 12),
    ]))
    table.wrapOn(c, width, height)
    table.drawOn(c, 310, height - 330 - 20 * len(violations))

    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(secondary_color)
    c.drawString(320, height - 360 - 20 * len(violations), f"Total Fine Amount: ₹{fine_amount}")

    qr_img = generate_qr_code(challan_id)
    qr_buffer = BytesIO()
    qr_img.save(qr_buffer)
    qr_buffer.seek(0)
    c.drawImage(ImageReader(qr_buffer), width - 160, height - 320, width=120, height=120)

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
    },
    "helmet": {
        "fine": 0,
        "icon": "✅",
        "severity": "info",
        "desc": "Helmet worn - no violation."
    },
    "rider": {
        "fine": 0,
        "icon": "👤",
        "severity": "info",
        "desc": "Rider detected."
    }
}

def main():
    st.set_page_config(page_title="Traffic Violation Detection & E-Challan", layout="wide")

    # Sidebar
    st.sidebar.title("Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("Traffic Violation Detection app with YOLO model and OCR for number plate extraction.\n\nBuilt by Wajahat.")

    st.title("Traffic Violation Detection & E-Challan Generator")

    uploaded_file = st.file_uploader("Upload Vehicle Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns([2, 3])

        with col1:
            st.image(temp_path, caption="Uploaded Vehicle Image", use_column_width=True)

        with col2:
            with st.spinner("Extracting number plate and OCR bounding boxes..."):
                plate, ocr_img = extract_number_plate(temp_path, conf_threshold=conf_threshold)
            if ocr_img is not None:
                st.image(ocr_img, caption="OCR Bounding Boxes on Preprocessed Image", use_column_width=True)
            st.markdown(f"**Extracted Number Plate:** {plate or 'Not detected'}")

        with st.spinner("Analyzing for violations..."):
            violations_raw = analyze_violations(temp_path)

        # Remove duplicates, keep order
        seen = set()
        violations_filtered = []
        for v in violations_raw:
            if v not in seen:
                seen.add(v)
                violations_filtered.append(v)

        violations = violations_filtered

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
