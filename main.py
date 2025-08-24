from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load your YOLO model
model_path = os.getenv("MODEL_PATH", "models/ModelTrain.pt")
model = YOLO(model_path)

@app.route("/", methods=["GET"])
def home():
    return "YOLO Flask App is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Run YOLO prediction
    results = model(file_path)
    predictions = results[0].boxes.xyxy.tolist()  # bounding boxes

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
