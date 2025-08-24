# main.py
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from PIL import Image
import io
import torch

app = FastAPI()

MODEL_PATH = "ModelTrain.pt"  # put model in same folder
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model once
model = YOLO(MODEL_PATH)
# Optionally set model.conf = 0.6  # default conf threshold you want

def read_imagefile(raw_bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    arr = np.array(image)[:, :, ::-1].copy()  # RGB->BGR
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = read_imagefile(contents)  # BGR numpy array

    # Run model (single image)
    results = model(img, imgsz=640, verbose=False)  # adjust imgsz if needed

    detections = []
    annotated = img.copy()
    names = model.model.names if hasattr(model, "model") else getattr(model, "names", {0:"0"})

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = names.get(cls, str(cls))
            detections.append({"label": label, "conf": conf, "bbox": [x1, y1, x2, y2]})
            color = (0,255,0) if label.lower()=="real" else (0,0,255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, buf = cv2.imencode('.jpg', annotated)
    jpg_b64 = base64.b64encode(buf).decode('utf-8')
    return {"detections": detections, "annotated_image": jpg_b64}
