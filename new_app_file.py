import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from PIL import Image, ImageOps
import io

app = Flask(__name__)

# Load model with better error handling
try:
    model = load_model('doclocked_model.h5')
    # Warm up the model
    dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)
    _ = model.predict(dummy_input)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

class_names = ['original', 'fake']

def preprocess_image(img):
    """Enhanced preprocessing pipeline"""
    # Convert to RGB if needed
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Resize with anti-aliasing
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Normalize and preprocess for EfficientNet (if using that architecture)
    img = preprocess_input(img.astype('float32'))
    
    return img

def predict_document_from_cv2(img):
    """Enhanced prediction function with better preprocessing"""
    try:
        # Enhanced preprocessing
        img = preprocess_image(img)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Predict with model
        prediction = model.predict(img, verbose=0)
        
        # Get results
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        # Apply confidence threshold (adjust based on your validation results)
        if confidence < 0.7:  # Example threshold
            predicted_class = "uncertain"
            
        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "error", 0.0

@app.route('/scan', methods=['POST'])
def scan():
    """Enhanced API endpoint with better error handling and validation"""
    # Check content type
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    
    # Validate input
    if 'image' not in data or not data['image']:
        return jsonify({'error': 'No image provided'}), 400

    img_data = data['image']
    
    try:
        # Handle different image formats
        if img_data.startswith('data:image'):
            header, encoded = img_data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
        else:
            # Assume it's raw base64 without header
            img_bytes = base64.b64decode(img_data)
            
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Try multiple image decoding methods
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to PIL if OpenCV fails
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                img = np.array(pil_img)
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            except Exception as pil_e:
                return jsonify({'error': f'Image decoding failed: {str(pil_e)}'}), 400
        
        # Validate image
        if img is None or img.size == 0:
            return jsonify({'error': 'Invalid or empty image'}), 400
            
        # Predict
        predicted_class, confidence = predict_document_from_cv2(img)
        
        # Prepare response
        response = {
            'class': predicted_class,
            'confidence': confidence,
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Processing failed: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == "__main__":
    # Configure for production
    app.run(host='0.0.0.0', port=5000, threaded=True)
