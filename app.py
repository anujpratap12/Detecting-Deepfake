from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import traceback

app = Flask(__name__)

# === Device configuration ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# === Define class names (MUST match training order!)
class_names = ['fake', 'real']  # ⚠️ Change this if your training dataset uses a different folder order

# === Image transformation (same as used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Load model and weights ===
model = resnet18(weights=None)  # Don't use pretrained here
model.fc = nn.Linear(model.fc.in_features, len(class_names))

try:
    model.load_state_dict(torch.load("deepfake_detector.pth", map_location=device))
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:")
    traceback.print_exc()
    raise

model.to(device)
model.eval()

# === Prediction endpoint ===
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        print("❗ No image key in request.files")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        print(f"📷 Received image: {file.filename}")

        image = Image.open(file.stream).convert("RGB")
        print("🖼️ Image opened successfully")

        image = transform(image).unsqueeze(0).to(device)
        print("🔁 Image transformed")

        with torch.no_grad():
            outputs = model(image)
            print("✅ Model ran successfully")

            _, predicted = outputs.max(1)
            prediction = class_names[predicted.item()]
            print(f"🎯 Predicted: {prediction}")

        return jsonify({"prediction": prediction})

    except Exception as e:
        print("🔥 ERROR OCCURRED DURING PREDICTION:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Start server ===
if __name__ == "__main__":
    app.run(debug=True)
