import cv2
import numpy as np
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
from PIL import Image

def capture_image_from_camera(camera_index=0):
    """Capture image from webcam"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise IOError("Failed to capture image")
    
    return frame

def convert_cv2_to_django_file(cv2_image, filename='captured.jpg'):
    """Convert OpenCV image to Django UploadedFile"""
    image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image)
    
    buffer = BytesIO()
    img_pil.save(buffer, format='JPEG')
    
    return InMemoryUploadedFile(
        buffer,
        None,
        filename,
        'image/jpeg',
        buffer.getbuffer().nbytes,
        None
    )

def validate_captured_image(image):
    """Validate the captured image meets requirements"""
    if image is None:
        raise ValueError("No image data received")
    
    # Check image dimensions
    height, width = image.shape[:2]
    if height < 100 or width < 100:
        raise ValueError("Image resolution too small (min 100x100 pixels)")
    
    # Check if image contains a face
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        raise ValueError("No face detected in the image")
    
    return True