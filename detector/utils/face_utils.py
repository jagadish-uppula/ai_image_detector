from mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import os
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

# Initialize models
try:
    detector = MTCNN(
        min_face_size=20,
        steps_threshold=[0.6, 0.7, 0.7],
        scale_factor=0.8
    )
except Exception as e:
    logger.error(f"Error initializing MTCNN detector: {str(e)}")
    raise

# Custom layer for FaceNet
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class L2Normalize(Layer):
    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1)

# Load FaceNet model
try:
    facenet_model = load_model(
        os.path.join(settings.BASE_DIR, 'models', 'facenet_keras.h5'),
        custom_objects={'L2Normalize': L2Normalize},
        compile=False
    )
except Exception as e:
    logger.error(f"Error loading FaceNet model: {str(e)}")
    raise

def extract_face(image_path, required_size=(160, 160)):
    """Extract face from image with enhanced error handling"""
    try:
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            # Handle file object
            image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not read image file")
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = detector.detect_faces(image)
        if not results:
            raise ValueError("No faces detected in the image")
        
        # Get the largest face
        largest_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
        x1, y1, width, height = largest_face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        # Add 20% margin around the face
        margin_w = int(width * 0.2)
        margin_h = int(height * 0.2)
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(image.shape[1], x2 + margin_w)
        y2 = min(image.shape[0], y2 + margin_h)
        
        # Extract and resize face
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, required_size)
        return face
    
    except Exception as e:
        logger.error(f"Face extraction error: {str(e)}")
        raise ValueError(f"Face extraction failed: {str(e)}")

def extract_face_embeddings(image_path):
    """Extract face embeddings with robust error handling"""
    if facenet_model is None:
        raise ValueError("FaceNet model not loaded")
    
    try:
        # Handle both file paths and file objects
        if isinstance(image_path, str):
            # It's a file path
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image file")
        else:
            # It's a file object
            image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = detector.detect_faces(image)
        if not results:
            raise ValueError("No faces detected in the image")
        
        # Get the largest face
        largest_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
        x1, y1, width, height = largest_face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        # Add margin
        margin_w = int(width * 0.2)
        margin_h = int(height * 0.2)
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(image.shape[1], x2 + margin_w)
        y2 = min(image.shape[0], y2 + margin_h)
        
        # Extract and resize face
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        
        # Preprocess face
        face = face.astype('float32')
        mean = np.mean(face)
        std = np.std(face)
        std_adj = np.maximum(std, 1.0/np.sqrt(face.size))
        face = (face - mean) / std_adj
        
        # Get embedding
        face = np.expand_dims(face, axis=0)
        embedding = facenet_model.predict(face)
        return embedding[0]
    
    except Exception as e:
        print(f"Embedding extraction error: {str(e)}")
        raise ValueError(f"Could not extract face features: {str(e)}")

def compare_faces(embedding1, embedding2, threshold=0.45):  # Lower threshold for stricter matching
    """
    Compare face embeddings with adjustable threshold
    Returns: (similarity_score, is_match)
    """
    try:
        if embedding1 is None or embedding2 is None:
            return 0.0, False
        
        # Ensure embeddings are normalized
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        similarity = np.dot(embedding1, embedding2)
        
        # More strict matching criteria
        is_match = similarity > threshold
        
        return similarity, is_match
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}")
        return 0.0, False