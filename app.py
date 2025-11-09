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
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, Image as RLImage
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

    # Colors and styles
    primary_color = colors.HexColor("#0B3D91")  # Deep Blue
    secondary_color = colors.HexColor("#F24C00")  # Orange/red for highlights
    styles = getSampleStyleSheet()
    styleN = styles["Normal"]
    styleB = styles["Heading2"]
    styleB.alignment = TA_CENTER

    # Header
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

    # Vehicle image
    if os.path.exists(vehicle_image_path):
        max_width = 240
        max_height = 180
        c.drawString(50, height - 170, "Vehicle Image:")
        c.drawImage(ImageReader(vehicle_image_path), 50, height - 350, width=max_wi_
