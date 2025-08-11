from flask import Flask, render_template, request, jsonify, Response
import os
import logging
import sys
import gc  # Garbage collection for memory management

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Lazy loading - only import heavy libraries when needed
cv2 = None
mp = None
np = None
pickle = None
Image = None
base64 = None
io = None

def load_heavy_imports():
    """Load heavy libraries only when needed"""
    global cv2, mp, np, pickle, Image, base64, io
    
    if cv2 is None:
        try:
            import cv2
            logger.info("✅ OpenCV loaded")
        except ImportError as e:
            logger.error(f"❌ Failed to import OpenCV: {e}")
            return False
    
    if mp is None:
        try:
            import mediapipe as mp
            logger.info("✅ MediaPipe loaded")
        except ImportError as e:
            logger.error(f"❌ Failed to import MediaPipe: {e}")
            return False
    
    if np is None:
        try:
            import numpy as np
            logger.info("✅ NumPy loaded")
        except ImportError as e:
            logger.error(f"❌ Failed to import NumPy: {e}")
            return False
            
    if pickle is None:
        try:
            import pickle
            logger.info("✅ Pickle loaded")
        except ImportError as e:
            logger.error(f"❌ Failed to import Pickle: {e}")
            return False
    
    if Image is None:
        try:
            from PIL import Image
            logger.info("✅ PIL loaded")
        except ImportError as e:
            logger.error(f"❌ Failed to import PIL: {e}")
            return False
    
    if base64 is None:
        try:
            import base64
            import io
            logger.info("✅ Base64 and IO loaded")
        except ImportError as e:
            logger.error(f"❌ Failed to import base64/io: {e}")
            return False
    
    return True

# Global variables - initialized on first use
model = None
hands = None
mp_hands = None
processor = None

def initialize_ml_components():
    """Initialize ML components with memory optimization"""
    global model, hands, mp_hands, processor
    
    if not load_heavy_imports():
        return False
    
    # Load model
    try:
        model_path = "sign_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning(f"❌ Model file not found at {model_path}")
            model = None
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        model = None
    
    # Initialize MediaPipe with memory-efficient settings
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,  # More memory efficient
            max_num_hands=1,         # Reduced from 2
            min_detection_confidence=0.7,  # Higher threshold
            min_tracking_confidence=0.5
        )
        logger.info("✅ MediaPipe initialized with optimized settings")
    except Exception as e:
        logger.error(f"❌ Error initializing MediaPipe: {str(e)}")
        hands = None
        return False
    
    # Initialize processor
    processor = SignLanguageProcessor()
    
    # Force garbage collection
    gc.collect()
    
    return True

class SignLanguageProcessor:
    def __init__(self):
        self.prev_prediction = ""
        self.current_letter = ""
        self.predicted_word = ""
        self.stable_count = 0
        self.stable_threshold = 5
        
    def reset(self):
        self.prev_prediction = ""
        self.current_letter = ""
        self.predicted_word = ""
        self.stable_count = 0

# Dummy letters for testing when model is not available
dummy_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'O', 'V', 'Y']

@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_sign():
    # Initialize components on first request
    if hands is None:
        logger.info("Initializing ML components...")
        if not initialize_ml_components():
            return jsonify({'error': 'Failed to initialize ML components'}), 500
    
    try:
        # Get image data from request
        data = request.get_json()
        if not data or 'image' not in data:
            logger.error("No image data received")
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image'].split(',')[1]
        
        # Process image with memory management
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize image to reduce memory usage
            if image.size[0] > 640:
                image = image.resize((640, 480))
            
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return jsonify({'error': 'Invalid image data'}), 400
        
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
                if model is not None:
                    try:
                        landmark_data = np.array(landmark_data).reshape(1, -1)
                        prediction = model.predict(landmark_data)[0]
                        logger.debug(f"Model prediction: {prediction}")
                    except Exception as e:
                        logger.error(f"Error making prediction: {str(e)}")
                        prediction = "ERROR"
                else:
                    # Use dummy prediction for testing
                    import random
                    prediction = random.choice(dummy_letters)
                    logger.debug(f"Dummy prediction: {prediction}")
                
                # Process prediction with stability check
                if prediction == processor.current_letter:
                    processor.stable_count += 1
                else:
                    processor.current_letter = prediction
                    processor.stable_count = 0
                
                # Add to word if stable enough
                if (processor.stable_count >= processor.stable_threshold and 
                    prediction != processor.prev_prediction):
                    processor.predicted_word += prediction
                    processor.prev_prediction = prediction
                    processor.stable_count = 0
                
                break  # Only process first hand
        
        # Clean up memory
        del opencv_image, rgb_image, image
        if 'result' in locals():
            del result
        gc.collect()
        
        response_data = {
            'prediction': prediction,
            'word': processor.predicted_word,
            'landmarks_detected': landmarks_detected,
            'stable_count': processor.stable_count,
            'model_available': model is not None
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in predict_sign: {str(e)}")
        # Force cleanup on error
        gc.collect()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/clear_word', methods=['POST'])
def clear_word():
    try:
        if processor:
            processor.predicted_word = ""
            processor.prev_prediction = ""
            processor.stable_count = 0
        logger.info("Word cleared")
        return jsonify({'status': 'cleared'})
    except Exception as e:
        logger.error(f"Error clearing word: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/speak_word', methods=['POST'])
def speak_word():
    try:
        word = processor.predicted_word if processor else ""
        logger.info(f"Speaking word: {word}")
        return jsonify({'word': word})
    except Exception as e:
        logger.error(f"Error getting word to speak: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    status = {
        'status': 'healthy',
        'components_loaded': hands is not None,
        'model_loaded': model is not None,
        'python_version': sys.version,
        'memory_optimized': True
    }
    logger.info(f"Health check: {status}")
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}")
    logger.info("Using lazy loading for memory optimization")
    app.run(host='0.0.0.0', port=port, debug=False)