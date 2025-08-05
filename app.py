# from flask import Flask, render_template, request, jsonify, Response
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import base64
# import io
# from PIL import Image
# import json
# import os

# app = Flask(__name__)

# # Load the trained model
# try:
#     with open("sign_model.pkl", "rb") as f:
#         model = pickle.load(f)
#     print("✅ Model loaded successfully")
# except FileNotFoundError:
#     print("❌ Model file not found. Please ensure sign_model.pkl exists.")
#     model = None

# # Initialize MediaPipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils

# class SignLanguageProcessor:
#     def __init__(self):
#         self.prev_prediction = ""
#         self.current_letter = ""
#         self.predicted_word = ""
#         self.stable_count = 0
#         self.stable_threshold = 5  # Reduced for web app responsiveness
        
#     def reset(self):
#         self.prev_prediction = ""
#         self.current_letter = ""
#         self.predicted_word = ""
#         self.stable_count = 0

# # Global processor instance
# processor = SignLanguageProcessor()

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_sign():
#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     try:
#         # Get image data from request
#         data = request.get_json()
#         image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
#         # Decode base64 image
#         image_bytes = base64.b64decode(image_data)
#         image = Image.open(io.BytesIO(image_bytes))
        
#         # Convert to OpenCV format
#         opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
#         # Process with MediaPipe
#         rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
#         result = hands.process(rgb_image)
        
#         prediction = None
#         landmarks_detected = False
        
#         if result.multi_hand_landmarks:
#             landmarks_detected = True
#             for hand_landmarks in result.multi_hand_landmarks:
#                 # Extract landmark data
#                 landmark_data = []
#                 for lm in hand_landmarks.landmark:
#                     landmark_data.extend([lm.x, lm.y, lm.z])
                
#                 # Make prediction
#                 landmark_data = np.array(landmark_data).reshape(1, -1)
#                 prediction = model.predict(landmark_data)[0]
                
#                 # Process prediction with stability check
#                 if prediction == processor.current_letter:
#                     processor.stable_count += 1
#                 else:
#                     processor.current_letter = prediction
#                     processor.stable_count = 0
                
#                 # Add to word if stable enough
#                 if processor.stable_count >= processor.stable_threshold and prediction != processor.prev_prediction:
#                     processor.predicted_word += prediction
#                     processor.prev_prediction = prediction
#                     processor.stable_count = 0
                
#                 break  # Only process first hand
        
#         return jsonify({
#             'prediction': prediction,
#             'word': processor.predicted_word,
#             'landmarks_detected': landmarks_detected,
#             'stable_count': processor.stable_count
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/clear_word', methods=['POST'])
# def clear_word():
#     processor.predicted_word = ""
#     processor.prev_prediction = ""
#     processor.stable_count = 0
#     return jsonify({'status': 'cleared'})

# @app.route('/speak_word', methods=['POST'])
# def speak_word():
#     # For web deployment, return the word to be spoken by browser's speech synthesis
#     return jsonify({'word': processor.predicted_word})

# @app.route('/health')
# def health_check():
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': model is not None
#     })

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
#     app.run(host='0.0.0.0', port=port, debug=debug_mode)
from flask import Flask, render_template, request, jsonify
import base64
import io
from PIL import Image
import json
import os
import logging
import sys
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Lightweight processor without heavy ML libraries
class LightweightSignProcessor:
    def __init__(self):
        self.prev_prediction = ""
        self.current_letter = ""
        self.predicted_word = ""
        self.stable_count = 0
        self.stable_threshold = 3
        # Simple demo letters
        self.demo_letters = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']
        self.demo_index = 0
        
    def reset(self):
        self.prev_prediction = ""
        self.current_letter = ""
        self.predicted_word = ""
        self.stable_count = 0
        self.demo_index = 0
    
    def get_demo_prediction(self):
        # Simulate sign recognition with demo sequence
        if self.demo_index < len(self.demo_letters):
            prediction = self.demo_letters[self.demo_index]
            self.demo_index += 1
        else:
            # Random letter after demo
            prediction = random.choice(['A', 'B', 'C', 'D', 'E'])
        return prediction

processor = LightweightSignProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sign():
    try:
        # Get image data (we'll ignore it for demo)
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Simulate processing delay
        import time
        time.sleep(0.1)
        
        # Simulate hand detection (80% chance)
        landmarks_detected = random.random() > 0.2
        
        prediction = None
        if landmarks_detected:
            # Get demo prediction
            prediction = processor.get_demo_prediction()
            
            # Process with stability
            if prediction == processor.current_letter:
                processor.stable_count += 1
            else:
                processor.current_letter = prediction
                processor.stable_count = 0
            
            # Add to word if stable
            if (processor.stable_count >= processor.stable_threshold and 
                prediction != processor.prev_prediction):
                processor.predicted_word += prediction
                processor.prev_prediction = prediction
                processor.stable_count = 0
        
        return jsonify({
            'prediction': prediction,
            'word': processor.predicted_word,
            'landmarks_detected': landmarks_detected,
            'stable_count': processor.stable_count,
            'demo_mode': True
        })
        
    except Exception as e:
        logger.error(f"Error in predict_sign: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_word', methods=['POST'])
def clear_word():
    processor.predicted_word = ""
    processor.prev_prediction = ""
    processor.stable_count = 0
    return jsonify({'status': 'cleared'})

@app.route('/speak_word', methods=['POST'])
def speak_word():
    return jsonify({'word': processor.predicted_word})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'mode': 'lightweight_demo',
        'memory_usage': 'low'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
