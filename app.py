import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('doclocked_model.h5')
class_names = ['original', 'fake']

def predict_document_from_cv2(img):
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

@app.route('/scan', methods=['POST'])
def scan():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    img_data = data['image']

    try:
        header, encoded = img_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': 'Invalid image data: ' + str(e)}), 400

    if img is None:
        return jsonify({'error': 'Image decoding failed'}), 400

    predicted_class, confidence = predict_document_from_cv2(img)

    return jsonify({'class': predicted_class, 'confidence': confidence})

if __name__ == "__main__":
    app.run(debug=True)
