import os
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
MODEL_PATH = "traffic_sign_model.h5"
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Label names
labels = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "Speed Limit 60",
    4: "Speed Limit 70",
    5: "Speed Limit 80",
    6: "End of Speed Limit 80",
    7: "Speed Limit 100",
    8: "Speed Limit 120",
    9: "No Overtaking",
    10: "No Overtaking (Trucks)",
    11: "Right-of-way at Intersection",
    12: "Priority Road",
    13: "Yield",
    14: "Stop",
    15: "No Traffic Both Ways",
    16: "No Trucks",
    17: "No Entry",
    18: "Danger",
    19: "Bend Left",
    20: "Bend Right",
    21: "Bend",
    22: "Uneven Road",
    23: "Slippery Road",
    24: "Road Narrows",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Children Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End of All Restrictions",
    33: "Turn Right",
    34: "Turn Left",
    35: "Ahead Only",
    36: "Go Straight or Right",
    37: "Go Straight or Left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout",
    41: "End of No Overtaking",
    42: "End of No Overtaking (Trucks)"
}

def preprocess_image(img):
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.reshape(1, 32, 32, 3)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read the file directly into a numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Preprocess and predict
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        class_id = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        result = labels.get(class_id, "Unknown Sign")
        
        return jsonify({
            'class_id': int(class_id),
            'label': result,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
