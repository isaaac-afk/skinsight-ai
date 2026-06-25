"""
SkinSight AI — Flask backend.

Serves the trained EfficientNet-B0 classifier with Grad-CAM explainability.
Endpoints:
    GET  /              -> the demo page (static/index.html)
    POST /api/analyze   -> multipart image upload; returns predictions,
                           a base64 Grad-CAM overlay, and a focus/trust score.

Run:
    python app.py
    # then open http://127.0.0.1:5000

EDUCATIONAL DEMO ONLY — not a medical device, outputs are not diagnoses.
"""

import base64
import io
import traceback

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

from skinsight_gradcam import GradCAM

app = Flask(__name__, static_folder="static", static_url_path="")

# Human-readable names for the ISIC 2019 class codes.
CLASS_FULL_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
}

# Load the model once at startup (not per-request).
print("Loading model + Grad-CAM engine...")
explainer = GradCAM()
print(f"Ready. Classes: {explainer.classes}")


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Attach a file under the 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read that file as an image."}), 400

    try:
        result = explainer.explain(img)

        # Attach full names and shape the response for the front-end.
        predictions = [
            {
                "code": p["label"],
                "name": CLASS_FULL_NAMES.get(p["label"], p["label"]),
                "confidence": p["confidence"],
            }
            for p in result["predictions"]
        ]

        return jsonify({
            "predictions": predictions,
            "top": {
                "code": result["top_label"],
                "name": CLASS_FULL_NAMES.get(result["top_label"], result["top_label"]),
            },
            "focus_score": result["focus_score"],
            "original": pil_to_base64(img),
            "overlay": pil_to_base64(result["overlay"]),
            "disclaimer": result["disclaimer"],
        })
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Analysis failed. See server console for details."}), 500


if __name__ == "__main__":
    # threaded=False keeps Grad-CAM's backward pass from racing on a shared model.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=False)
