import gradio as gr
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
import tempfile
from ultralytics import YOLO

# -------------------- Paths & Config --------------------
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Update these to your actual model paths
PLATE_MODEL_PATH = BASE_DIR / "license_plate_detector.pt"
HELMET_MODEL_PATH = BASE_DIR / "best.pt"

# -------------------- Load OCR & Models --------------------
reader = easyocr.Reader(['en'], gpu=False)
plate_model = YOLO(PLATE_MODEL_PATH)
helmet_model = YOLO(HELMET_MODEL_PATH)

# -------------------- Fine Amounts --------------------
FINE_AMOUNTS = {
    "without helmet": 500,
    "no seatbelt": 300,
    "triple riding": 700,
    # Add other violations and fines as per your model/class names
}

# -------------------- Helper Functions --------------------
def extract_number_plate_from_image(image_path: Path) -> str | None:
    results = plate_model.predict(source=str(image_path), conf=0.3, verbose=False)
    if not results or len(results) == 0:
        return None

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None

    img = cv2.imread(str(image_path))
    x1, y1, x2, y2 = boxes[0].astype(int)
    cropped = img[y1:y2, x1:x2]

    if cropped.size == 0:
        return None

    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    ocr_results = reader.readtext(gray_cropped)
    texts = [text for _, text, conf in ocr_results if conf > 0.4]
    return " ".join(texts).strip() if texts else None

def analyze_helmet_violations(image_path: Path) -> list[str]:
    results = helmet_model.predict(source=str(image_path), conf=0.3, verbose=False)
    violations = []
    if results and len(results) > 0:
        for r in results:
            for cls in r.boxes.cls.cpu().numpy():
                class_name = helmet_model.names[int(cls)]
                violations.append(class_name)
    return violations

def generate_challan_pdf(vehicle_number: str, violations: list[str], fine_amount: int, challan_id: str) -> BytesIO:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Traffic Violation E-Challan")

    c.setFont("Helvetica", 14)
    c.drawString(50, height - 100, f"Challan ID: {challan_id}")
    c.drawString(50, height - 130, f"Vehicle Number: {vehicle_number or 'UNKNOWN'}")
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

# -------------------- Gradio App Logic --------------------
def process(image):
    # Save input PIL image to temp path
    temp_path = TEMP_DIR / "temp_image.jpg"
    image_np = np.array(image)
    cv2.imwrite(str(temp_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    plate_text = extract_number_plate_from_image(temp_path) or "Not detected"
    violations = analyze_helmet_violations(temp_path)

    total_fine = sum(FINE_AMOUNTS.get(v.lower(), 0) for v in violations)

    if violations:
        challan_id = generate_challan_id()
        pdf_buffer = generate_challan_pdf(plate_text, violations, total_fine, challan_id)

        # Write PDF to a temp file for Gradio to serve
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_buffer.getvalue())
            pdf_file_path = tmp_file.name
    else:
        pdf_file_path = None

    return plate_text, "\n".join(violations) if violations else "No violations detected.", total_fine, pdf_file_path

# -------------------- Gradio UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🚦 Traffic Violation Detection & E-Challan Generator")

    with gr.Row():
        image_input = gr.Image(label="Upload Vehicle Image", type="pil", interactive=True)

    with gr.Row():
        plate_output = gr.Textbox(label="Extracted Number Plate", interactive=False)
        violations_output = gr.Textbox(label="Detected Violations", interactive=False)
        fine_output = gr.Number(label="Total Fine Amount (₹)", interactive=False)

    pdf_download = gr.File(label="Download Challan PDF", file_types=[".pdf"])

    btn = gr.Button("Analyze Image")

    btn.click(process, inputs=image_input, outputs=[plate_output, violations_output, fine_output, pdf_download])

if __name__ == "__main__":
    demo.launch()
