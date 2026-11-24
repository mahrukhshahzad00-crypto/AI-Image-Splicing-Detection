import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, render_template
import segmentation_models_pytorch as smp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "trainingresults/best_model.pth"
IMG_SIZE   = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded on {DEVICE}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def preprocess(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


def to_b64(pil_img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def colormap_mask(prob: np.ndarray) -> Image.Image:
    """Convert float32 [0,1] prob map to a red-heatmap RGBA image."""
    h, w = prob.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = (prob * 255).astype(np.uint8)          # red channel
    rgba[..., 3] = (prob * 200).astype(np.uint8)          # alpha
    return Image.fromarray(rgba, "RGBA")


def overlay_mask(orig: Image.Image, prob: np.ndarray) -> Image.Image:
    base = orig.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR).convert("RGBA")
    heat = colormap_mask(prob)
    return Image.alpha_composite(base, heat).convert("RGB")


@torch.no_grad()
def run_inference(pil_img: Image.Image):
    tensor = preprocess(pil_img)
    with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
        logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().float().numpy()   # (256, 256)

    # Forgery score: mean of top-1% pixel probabilities
    flat = prob.flatten()
    k = max(1, len(flat) // 100)
    score = float(np.partition(flat, -k)[-k:].mean())

    return prob, score

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        pil_img = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Cannot read image"}), 400

    prob, score = run_inference(pil_img)

    # Binary mask at 0.5 threshold
    binary = (prob > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(binary, "L")

    # Probability heatmap (grayscale)
    heatmap_img = Image.fromarray((prob * 255).astype(np.uint8), "L")

    # Overlay
    overlay_img = overlay_mask(pil_img, prob)

    # Original resized
    orig_resized = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    return jsonify({
        "score":    round(score * 100, 1),
        "original": to_b64(orig_resized),
        "mask":     to_b64(mask_img),
        "heatmap":  to_b64(heatmap_img),
        "overlay":  to_b64(overlay_img),
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
