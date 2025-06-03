import os
from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import base64 # To encode/decode images for display

# Add this line:
import torch.nn.functional as F

# Import your CNNModel class from cnn_model.py
from cnn_model import CNNModel

app = Flask(__name__)

# --- Configuration ---
# Path to your saved model weights
MODEL_PATH = 'cnn_model_weights.pth'
# Determine the device (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load the trained model ---
# Instantiate your CNNModel
model = CNNModel()

# Load the saved state dictionary
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set the model to evaluation mode
    model.to(DEVICE) # Move model to the appropriate device
    print(f"CNN Model loaded successfully from {MODEL_PATH} on {DEVICE}")
except FileNotFoundError:
    print(f"Error: Model weights file not found at {MODEL_PATH}.")
    print("Please ensure you have trained the model and saved its weights.")
    print("Run `python cnn_model.py` (or `main.py` if you save there) first.")
    exit() # Exit if model not found

# --- Define transformations (MUST match data_loader.py) ---
# Assuming you resized images to 64x64 in data_loader.py
inference_transform = transforms.Compose([
    transforms.Resize((64, 64)), # Ensure this matches your training data size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Routes ---

@app.route('/')
def index():
    """Renders the main HTML page for image upload."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, runs prediction, and returns result."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            # Read the image file from the request
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB') # Ensure RGB

            # Apply transformations
            input_tensor = inference_transform(img)
            input_batch = input_tensor.unsqueeze(0) # Add a batch dimension

            # Move input to the same device as the model
            input_batch = input_batch.to(DEVICE)

            # Make prediction
            with torch.no_grad():
                output = model(input_batch)
                probabilities = F.softmax(output, dim=1) # Get probabilities
                _, predicted_class = torch.max(output, 1) # Get the predicted class index

            # Map class index to label
            labels = {0: 'Normal', 1: 'Pneumonia'}
            prediction_label = labels[predicted_class.item()]
            
            # Get probability of the predicted class
            confidence = probabilities[0, predicted_class.item()].item() * 100

            # Encode image for display on the frontend (optional, but nice)
            img_io = io.BytesIO()
            img.save(img_io, format='PNG') # Save as PNG or JPEG
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            return jsonify({
                'prediction': prediction_label,
                'confidence': f"{confidence:.2f}%",
                'image': f"data:image/png;base64,{img_base64}" # Data URL for image
            })

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Ensure your data_dir path is correct for the data_loader used by cnn_model.py
    # if you ran that directly to generate the pth file.
    # For running the Flask app, no data_dir is directly needed here unless you re-train.
    
    print("Starting Flask application...")
    app.run(debug=True, port=5000) # Run in debug mode (auto-reloads on code changes)