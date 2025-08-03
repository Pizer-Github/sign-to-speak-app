from flask import Flask, render_template, request, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import io
from PIL import Image
import json
import os

app = Flask(__name__)

# Load the trained model
try:
    with open("sign_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found. Please ensure sign_model.pkl exists.")
    model = None

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

class SignLanguageProcessor:
    def __init__(self):
        self.prev_prediction = ""
        self.current_letter = ""
        self.predicted_word = ""
        self.stable_count = 0
        self.stable_threshold = 5  # Reduced for web app responsiveness
        
    def reset(self):
        self.prev_prediction = ""
        self.current_letter = ""
        self.predicted_word = ""
        self.stable_count = 0

# Global processor instance
processor = SignLanguageProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sign():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process with MediaPipe
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)
        
        prediction = None
        landmarks_detected = False
        
        if result.multi_hand_landmarks:
            landmarks_detected = True
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract landmark data
                landmark_data = []
                for lm in hand_landmarks.landmark:
                    landmark_data.extend([lm.x, lm.y, lm.z])
                
                # Make prediction
                landmark_data = np.array(landmark_data).reshape(1, -1)
                prediction = model.predict(landmark_data)[0]
                
                # Process prediction with stability check
                if prediction == processor.current_letter:
                    processor.stable_count += 1
                else:
                    processor.current_letter = prediction
                    processor.stable_count = 0
                
                # Add to word if stable enough
                if processor.stable_count >= processor.stable_threshold and prediction != processor.prev_prediction:
                    processor.predicted_word += prediction
                    processor.prev_prediction = prediction
                    processor.stable_count = 0
                
                break  # Only process first hand
        
        return jsonify({
            'prediction': prediction,
            'word': processor.predicted_word,
            'landmarks_detected': landmarks_detected,
            'stable_count': processor.stable_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_word', methods=['POST'])
def clear_word():
    processor.predicted_word = ""
    processor.prev_prediction = ""
    processor.stable_count = 0
    return jsonify({'status': 'cleared'})

@app.route('/speak_word', methods=['POST'])
def speak_word():
    # For web deployment, return the word to be spoken by browser's speech synthesis
    return jsonify({'word': processor.predicted_word})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
