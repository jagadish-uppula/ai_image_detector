from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from tensorflow.keras.applications.xception import preprocess_input

# Load Xception model
xception_model = load_model(
    os.path.join('models', 'xception.h5'),
    compile=False
)

def preprocess_image(image_path, target_size=(299, 299)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
            
        # Convert to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        
        # Use Xception's specific preprocessing
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        raise

def predict_ai_generated(image_path):
    try:
        # Handle both file paths and file objects
        if isinstance(image_path, str):
            # It's a file path
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")
        else:
            # It's a file object - read directly
            img = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        
        # Preprocess and predict
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        prediction = xception_model.predict(img, verbose=0)
        confidence = float(prediction[0][0])
        
        # Adjusted threshold
        is_ai = confidence > 0.85
        return is_ai, confidence if is_ai else 1 - confidence
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise