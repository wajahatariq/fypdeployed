import streamlit as st
import os
import cv2
import numpy as np
import easyocr
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import random
import string
from ultralytics import YOLO
import qrcode
from PIL import Image
from pathlib import Path

# Your existing helper functions here...
# (preprocess_image, extract_number_plate, analyze_violations, generate_qr_code, generate_challan_pdf, generate_challan_id, VIOLATION_DETAILS)

# Load model and EasyOCR reader only once (omitted for brevity)


def main():
    st.set_page_config(page_title="Traffic Violation Detection & E-Challan", layout="centered")

    # Load and apply CSS if exists
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        css_content = css_path.read_text()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    st.title("Traffic Violation Detection & E-Challan Generator")

    # Add slider in sidebar for OCR confidence threshold
    conf_threshold = st.sidebar.slider(
        "OCR Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Adjust confidence level for OCR text detection"
    )

    uploaded_file = st.file_uploader("Upload Vehicle Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)

        with col1:
            st.image(temp_path, caption="Uploaded Vehicle Image", use_column_width=True)

        with col2:
            with st.spinner("Extracting number plate and OCR bounding boxes..."):
                plate, ocr_img = extract_number_plate(temp_path, conf_threshold=conf_threshold)
            if ocr_img is not None:
                st.image(ocr_img, caption="OCR Bounding Boxes on Preprocessed Image", use_column_width=True)
            st.markdown(f"**Extracted Number Plate:** {plate or 'Not detected'}")

        # Section: Violation Detection
        st.header("Violation Detection Results")

        with st.spinner("Analyzing for violations..."):
            violations = analyze_violations(temp_path)

        if violations:
            for v in violations:
                details = VIOLATION_DETAILS.get(v.lower(), {"fine": 0, "icon": "❓", "severity": "info", "desc": ""})
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

        # Section: Challan PDF Generation
        if violations:
            st.header("Generate E-Challan PDF")
            challan_id = generate_challan_id()
            pdf_buffer = generate_challan_pdf(plate or "UNKNOWN", violations, total_fine, challan_id, temp_path)

            st.download_button(
                label="Download Challan PDF",
                data=pdf_buffer,
                file_name=f"challan_{challan_id}.pdf",
                mime="application/pdf"
            )

    else:
        st.info("Please upload a vehicle image to begin analysis.")

if __name__ == "__main__":
    main()
