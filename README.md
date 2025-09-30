# AI Image Detector

A Django-based web application for detecting AI-generated images and facial verification using deep learning models.

## Features

- AI Image Detection using Xception model
- Facial Verification using FaceNet
- Real-time camera capture
- User authentication system
- Security dashboard with analytics

## Technologies Used

- **Backend**: Django 4.2, Python 3.11
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **AI Models**: TensorFlow, OpenCV, FaceNet, Xception
- **Database**: SQLite

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-image-detector.git
cd ai-image-detector

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

python manage.py migrate

python manage.py createsuperuser

python manage.py runserver

