import torch
import torch.nn as nn
# No quantization imports needed anymore
import os
import json
import io
import secrets
import traceback
from werkzeug.utils import secure_filename
from torchvision import models
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template
# No FloatFunctional needed for FP32

# ==============================================================================
# Model Definition (Standard FP32 version)
# ==============================================================================

class LesionFocusAttention(nn.Module):
    """Standard FP32 Lesion Focus Attention module."""
    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # No FloatFunctional wrapper needed here

    def forward(self, x):
        attention_map = self.attention_conv(x)
        attention_map = self.sigmoid(attention_map)
        # Use standard multiplication for FP32
        return x * attention_map

def Agri_efficientnet(num_classes: int, use_lfa: bool = True):
    """Creates the FP32 EfficientNet-B0 model structure."""
    # Use standard EfficientNet B0 with default weights initially
    # (weights will be overwritten by loaded state_dict)
    model = models.efficientnet_b0(weights=None) # Start empty

    if use_lfa:
        last_conv_channels = model.features[-1][0].out_channels
        print(f">>> [App]: Inserting LesionFocusAttention (LFA)...")
        # Ensure LFA is placed correctly
        original_features = model.features
        model.features = nn.Sequential(
            original_features,
            LesionFocusAttention(last_conv_channels)
        )

    # Replace the classifier
    classifier_input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=classifier_input_features, out_features=num_classes)
    )
    return model

# --- Flask App Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global Variables ---
MODEL = None
CLASS_NAMES = []
SOLUTIONS = {}
DEVICE = torch.device("cpu") # Inference runs on CPU

# --- Image Preprocessing ---
inference_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def transform_image(image_bytes: bytes) -> torch.Tensor:
    """Transforms image bytes into a tensor."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return inference_transforms(image).unsqueeze(0)
    except Exception as e:
        print(f"Error transforming image: {e}")
        return None

# --- Load Standard FP32 Model ---

def load_all_data():
    """Loads metadata and the standard FP32 pruned model."""
    global MODEL, CLASS_NAMES, SOLUTIONS

    # File paths - Load the PRUNED FP32 model
    model_path = "pruned_finetuned_model.pth" # Load the pruned FP32 model
    class_names_path = "class_names.json"
    solution_path = "Solution.json"

    # Check existence
    for path in [model_path, class_names_path, solution_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Critical file missing: '{path}'")

    # Load metadata
    print("[INFO] Loading metadata (class names, solutions)...")
    with open(class_names_path, 'r', encoding='utf-8') as f: CLASS_NAMES = json.load(f)
    with open(solution_path, 'r', encoding='utf-8') as f: SOLUTIONS = json.load(f)

    # Load FP32 model
    try:
        print(f"[INFO] Loading FP32 model state_dict from '{model_path}'...")
        # 1. Create the standard FP32 model shell
        #    Make sure num_classes matches the saved model
        num_classes = len(CLASS_NAMES)
        model_shell = Agri_efficientnet(num_classes=num_classes)

        # 2. Load the weights
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        # Use strict=True as the architecture should match exactly now
        model_shell.load_state_dict(state_dict, strict=True)
        print("✅ Loaded FP32 state_dict successfully (strict=True).")

        # 3. Set to eval mode
        model_shell.eval()
        MODEL = model_shell
        print("✅ Deployment successful! FP32 model is ready.")

    except Exception as e:
        print(f"❌ Fatal error while loading FP32 model: {e}")
        traceback.print_exc()
        raise

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return "No file part", 400
    file = request.files['file']
    if file.filename == '': return "No file selected", 400

    if file:
        try:
            # Save, transform, predict
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            with open(filepath, 'rb') as f: image_bytes = f.read()
            tensor = transform_image(image_bytes)
            if tensor is None: return "Error processing image", 400

            with torch.no_grad():
                # Run standard FP32 inference
                outputs = MODEL(tensor.to(DEVICE))

            # Process output
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, top_catid = torch.topk(probabilities, 1)
            predicted_idx = top_catid[0].item()
            class_name = CLASS_NAMES[predicted_idx]
            solution = SOLUTIONS.get(class_name, {})

            # Render result
            return render_template('result.html', label=class_name, image_filename=filename, solution=solution)

        except Exception as e:
            print("="*50, "\n🔥🔥🔥 Prediction failed! 🔥🔥🔥")
            print(f"Error during prediction: {e}") # Log the specific error
            traceback.print_exc() # Print the full traceback
            print("="*50)
            return "Internal server error during prediction.", 500 # Generic message to browser

# --- Run Flask App ---
if __name__ == '__main__':
    try:
        load_all_data()
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Failed to start Flask app: {e}")

