from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.db.models import Avg, Count
from datetime import datetime, timedelta
from django.conf import settings
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import tempfile


from .forms import UserRegisterForm, LoginForm, ImageUploadForm, FaceVerificationForm, FaceRegistrationForm
from .models import AnalysisHistory, User
from .utils.face_utils import extract_face_embeddings, compare_faces
from .utils.camera_utils import validate_captured_image, convert_cv2_to_django_file, capture_image_from_camera
from .utils.model_utils import predict_ai_generated
from .utils.auth_utils import verify_login
from .utils.viz_utils import get_visualization_data

def home(request):
    return render(request, 'detector/home.html')

@login_required
def dashboard(request):
    total_analyses = AnalysisHistory.objects.filter(user=request.user).count()
    real_images = AnalysisHistory.objects.filter(user=request.user, is_ai_generated=False).count()
    ai_generated = AnalysisHistory.objects.filter(user=request.user, is_ai_generated=True).count()
    
    recent_analyses = AnalysisHistory.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    context = {
        'stats': {
            'total_analyses': total_analyses,
            'real_images': real_images,
            'ai_generated': ai_generated
        },
        'recent_analyses': recent_analyses
    }
    return render(request, 'detector/dashboard.html', context)

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {username}!')
                return redirect('dashboard')
            else:
                messages.error(request, 'Invalid username or password')
    else:
        form = LoginForm()
    return render(request, 'detector/login.html', {'form': form})

def user_register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                profile_pic = request.FILES['profile_picture']
                
                # Create a temporary file if the upload is in memory
                if hasattr(profile_pic, 'temporary_file_path'):
                    # File is already on disk
                    image_path = profile_pic.temporary_file_path()
                else:
                    # File is in memory - save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        for chunk in profile_pic.chunks():
                            tmp.write(chunk)
                        image_path = tmp.name
                
                try:
                    # 1. First check if image is AI-generated
                    try:
                        is_ai, confidence = predict_ai_generated(image_path)
                        if is_ai:
                            messages.error(request, 
                                f'Profile picture appears to be AI-generated (confidence: {confidence*100:.1f}%). Please use a real photo.')
                            if not hasattr(profile_pic, 'temporary_file_path'):
                                os.unlink(image_path)
                            return render(request, 'detector/register.html', {'form': form})
                    except Exception as e:
                        print(f"AI detection error: {str(e)}")
                        messages.error(request, 'Error analyzing image. Please try another photo.')
                        if not hasattr(profile_pic, 'temporary_file_path'):
                            os.unlink(image_path)
                        return render(request, 'detector/register.html', {'form': form})
                    
                    # 2. Extract face embeddings
                    try:
                        face_embedding = extract_face_embeddings(image_path)
                        if face_embedding is None:
                            raise ValueError("No face detected")
                    except Exception as e:
                        print(f"Face extraction error: {str(e)}")
                        messages.error(request, 'Could not detect a face in the image. Please try again with a clear photo of your face.')
                        if not hasattr(profile_pic, 'temporary_file_path'):
                            os.unlink(image_path)
                        return render(request, 'detector/register.html', {'form': form})
                    
                    # 3. Create user with face embedding
                    try:
                        user = form.save(commit=False)
                        user.profile_picture = profile_pic
                        user.face_embedding = pickle.dumps(face_embedding)
                        user.is_face_registered = True
                        user.save()
                        form.save_m2m()
                        
                        # Clean up temp file if we created one
                        if not hasattr(profile_pic, 'temporary_file_path'):
                            os.unlink(image_path)
                        
                        login(request, user)
                        messages.success(request, f'Registration successful! Welcome {user.username}!')
                        return redirect('dashboard')
                    
                    except Exception as e:
                        print(f"User creation error: {str(e)}")
                        messages.error(request, 'Error creating your account. Please try again.')
                        if not hasattr(profile_pic, 'temporary_file_path'):
                            os.unlink(image_path)
                        return render(request, 'detector/register.html', {'form': form})
                
                except Exception as e:
                    print(f"Registration error: {str(e)}")
                    messages.error(request, 'An error occurred during registration. Please try again.')
                    if not hasattr(profile_pic, 'temporary_file_path'):
                        os.unlink(image_path)
                    return render(request, 'detector/register.html', {'form': form})
            
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                messages.error(request, 'An unexpected error occurred. Please try again.')
                return render(request, 'detector/register.html', {'form': form})
    else:
        form = UserRegisterForm()
    
    return render(request, 'detector/register.html', {'form': form})

@login_required
def user_logout(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('home')

@login_required
def analyze(request):
    analyses = AnalysisHistory.objects.filter(user=request.user).order_by('-created_at')
    
    # Handle search and filtering
    query = request.GET.get('q')
    if query:
        analyses = analyses.filter(image__icontains=query)
    
    analysis_type = request.GET.get('type')
    if analysis_type == 'real':
        analyses = analyses.filter(is_ai_generated=False)
    elif analysis_type == 'ai':
        analyses = analyses.filter(is_ai_generated=True)
    
    sort = request.GET.get('sort', '-created_at')
    analyses = analyses.order_by(sort)
    
    return render(request, 'detector/analyze.html', {'analyses': analyses})

@login_required
def predict(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file
            uploaded_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            filepath = fs.path(filename)
            
            # Initialize result dictionary
            result = {
                'image_url': fs.url(filename),
                'created_at': datetime.now(),
                'is_ai': False,
                'confidence': 0.0,
                'face_detected': False,
                'face_match': False,
                'similarity_score': None,
                'has_stored_face': request.user.face_embedding is not None
            }
            
            try:
                # 1. AI Detection
                is_ai, confidence = predict_ai_generated(filepath)
                result['is_ai'] = is_ai
                result['confidence'] = float(confidence * 100)  # Convert to percentage
                
                # 2. Face Detection and Verification
                face_embedding = None
                try:
                    face_embedding = extract_face_embeddings(filepath)
                    result['face_detected'] = True
                    
                    if request.user.face_embedding is not None:
                        stored_embedding = pickle.loads(request.user.face_embedding)
                        similarity_score, is_match = compare_faces(stored_embedding, face_embedding, threshold=0.45)
                        result['similarity_score'] = float(similarity_score * 100)  # Convert to percentage
                        result['face_match'] = bool(is_match)
                except Exception as e:
                    print(f"Face detection failed: {str(e)}")
                
                # 3. Save to database
                analysis = AnalysisHistory(
                    user=request.user,
                    image=filename,
                    is_ai_generated=result['is_ai'],
                    confidence=result['confidence'],
                    similarity_score=result['similarity_score'],
                    face_embedding=pickle.dumps(face_embedding) if face_embedding is not None else None
                )
                analysis.save()
                
                # Add appropriate messages
                if result['face_detected']:
                    if result['has_stored_face']:
                        if result['face_match']:
                            messages.success(request, f'✅ Face verified! Similarity: {result["similarity_score"]:.1f}%')
                        else:
                            messages.warning(request, f'⚠️ Face not recognized. Similarity: {result["similarity_score"]:.1f}% (needs >45%)')
                    else:
                        messages.info(request, 'ℹ️ Face detected - register your face for verification')
                else:
                    messages.info(request, 'ℹ️ No face detected - verification not available')
                
                # AI detection feedback
                ai_status = "AI-Generated" if result['is_ai'] else "Real"
                confidence_level = "High" if result['confidence'] > 85 else "Medium" if result['confidence'] > 70 else "Low"
                messages.success(request, f'Image classified as {ai_status} with {confidence_level} confidence ({result["confidence"]:.1f}%)')
                
                return render(request, 'detector/predict.html', {'form': form, 'result': result})
            
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                messages.error(request, 'An error occurred during processing. Please try another image.')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect('predict')
    else:
        form = ImageUploadForm()
        
        # Handle re-analysis of existing image
        image_id = request.GET.get('image_id')
        if image_id:
            try:
                analysis = AnalysisHistory.objects.get(id=image_id, user=request.user)
                result = {
                    'image_url': analysis.image.url,
                    'created_at': analysis.created_at,
                    'is_ai': analysis.is_ai_generated,
                    'confidence': analysis.confidence,
                    'face_detected': analysis.face_embedding is not None,
                    'face_match': False,
                    'similarity_score': analysis.similarity_score,
                    'has_stored_face': request.user.face_embedding is not None
                }
                return render(request, 'detector/predict.html', {'form': form, 'result': result})
            except AnalysisHistory.DoesNotExist:
                pass
    
    return render(request, 'detector/predict.html', {'form': form})

@login_required
def visualize(request):
    # Get all visualization data
    data = get_visualization_data(request.user)
    ai_percentage = data.get('ai_percentage', 0)
    real_percentage = 100 - ai_percentage
    
    context = {
        'total_analyses': data['total_analyses'],
        'real_images': data['real_images'],
        'ai_generated': data['ai_generated'],
        'avg_confidence': data['avg_confidence'],
        'confidence_data': data['confidence_data'],
        'similarity_data': data['similarity_data'],
        'timeline_data': data['timeline_data'],
        'ai_percentage': ai_percentage,
        'real_percentage': real_percentage,
    }
    
    return render(request, 'detector/visualize.html', context)

@login_required
def update_profile_pic(request):
    if request.method == 'POST':
        profile_pic = request.FILES.get('profile_picture')
        if not profile_pic:
            messages.error(request, 'Please select an image')
            return redirect('profile')
        
        # Save the uploaded file temporarily
        fs = FileSystemStorage()
        filename = fs.save(profile_pic.name, profile_pic)
        filepath = fs.path(filename)
        
        try:
            # Verify if the image is AI-generated
            is_ai, confidence = predict_ai_generated(filepath)
            if is_ai:
                messages.error(request, 
                    f'Profile picture appears to be AI-generated (confidence: {confidence*100:.1f}%). Please use a real photo.')
                os.remove(filepath)
                return redirect('profile')
            
            # Extract face embeddings
            try:
                face_embedding = extract_face_embeddings(filepath)
            except Exception as e:
                messages.error(request, 'Could not detect a face in the image. Please try again.')
                os.remove(filepath)
                return redirect('profile')
            
            # Update user profile
            request.user.profile_picture = profile_pic
            request.user.set_face_embedding(face_embedding)
            request.user.save()
            
            # Clean up
            os.remove(filepath)
            
            messages.success(request, 'Profile picture updated successfully!')
            return redirect('profile')
        
        except Exception as e:
            print(f"Error updating profile picture: {str(e)}")
            messages.error(request, 'An error occurred. Please try again.')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect('profile')
    
    return redirect('profile')

@login_required
def profile(request):
    total_analyses = AnalysisHistory.objects.filter(user=request.user).count()
    ai_detected = AnalysisHistory.objects.filter(user=request.user, is_ai_generated=True).count()
    
    context = {
        'stats': {
            'total_analyses': total_analyses,
            'ai_detected': ai_detected
        }
    }
    return render(request, 'detector/profile.html', context)

@login_required
def security(request):
    if request.method == 'POST':
        form = FaceVerificationForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Process the captured image
                verification_img = request.FILES.get('verification_image')
                if not verification_img:
                    messages.error(request, 'Please capture or upload an image')
                    return render(request, 'detector/security.html', {'form': form})
                
                fs = FileSystemStorage()
                # Generate a unique filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"verification_{request.user.id}_{timestamp}.jpg"
                filename = fs.save(filename, verification_img)
                filepath = fs.path(filename)
                
                # Read the image with OpenCV for validation
                cv_image = cv2.imread(filepath)
                validate_captured_image(cv_image)
                
                # Verify if the image is AI-generated
                is_ai, confidence = predict_ai_generated(filepath)
                
                # Face verification
                face_embedding = extract_face_embeddings(filepath)
                similarity_score = 0
                is_match = False
                
                if request.user.is_face_registered:
                    stored_embedding = request.user.get_face_embedding()
                    if face_embedding is not None and stored_embedding is not None:
                        similarity_score, is_match = compare_faces(stored_embedding, face_embedding)
                        
                        # Additional check - ensure similarity is above threshold
                        if similarity_score < 0.3:
                            is_match = False
                
                # Prepare result
                result = {
                    'image_url': fs.url(filename),  # This will now point to the permanently stored file
                    'is_ai': is_ai,
                    'confidence': float(confidence * 100),
                    'face_detected': face_embedding is not None,
                    'face_match': is_match,
                    'similarity_score': float(similarity_score * 100) if similarity_score else None,
                    'has_stored_face': request.user.is_face_registered
                }
                
                # Don't delete the file - it's now permanently stored
                return render(request, 'detector/security.html', {
                    'form': form,
                    'result': result
                })
            
            except Exception as e:
                messages.error(request, f'Error during verification: {str(e)}')
                if 'filepath' in locals() and os.path.exists(filepath):
                    os.remove(filepath)
                return render(request, 'detector/security.html', {'form': form})
    else:
        form = FaceVerificationForm()
    
    # Add these stats to the context for the right sidebar
    total_analyses = AnalysisHistory.objects.filter(user=request.user).count()
    ai_detected = AnalysisHistory.objects.filter(user=request.user, is_ai_generated=True).count()
    face_verifications = AnalysisHistory.objects.filter(
        user=request.user, 
        similarity_score__gte=45
    ).count()
    
    return render(request, 'detector/security.html', {
        'form': form,
        'total_analyses': total_analyses,
        'ai_detected': ai_detected,
        'face_verifications': face_verifications
    })

def verify_face(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Process the uploaded image
            uploaded_file = request.FILES['image']
            
            # Verify if the image is AI-generated
            is_ai, confidence = predict_ai_generated(uploaded_file.temporary_file_path())
            if is_ai:
                return JsonResponse({
                    'is_verified': False,
                    'message': 'AI-generated image detected! Please use a real photo.'
                })
            
            # Extract face embeddings
            face_embedding = extract_face_embeddings(uploaded_file.temporary_file_path())
            
            # Check if face is properly detected
            if face_embedding is None:
                return JsonResponse({
                    'is_verified': False,
                    'message': 'No face detected. Please try again.'
                })
            
            # If this is a good face image
            return JsonResponse({
                'is_verified': True,
                'message': 'Face verified successfully!'
            })
            
        except Exception as e:
            return JsonResponse({
                'is_verified': False,
                'message': f'Verification error: {str(e)}'
            })
    
    return JsonResponse({
        'is_verified': False,
        'message': 'Invalid request'
    })

@login_required
def register_face(request):
    if request.method == 'POST':
        form = FaceRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['face_image']
            fs = FileSystemStorage()
            filename = fs.save(uploaded_file.name, uploaded_file)
            filepath = fs.path(filename)
            
            try:
                # Verify if the image is AI-generated
                is_ai, confidence = predict_ai_generated(filepath)
                if is_ai:
                    messages.error(request, 
                        f'Image appears to be AI-generated (confidence: {confidence*100:.1f}%). Please use a real photo.')
                    os.remove(filepath)
                    return render(request, 'detector/register_face.html', {'form': form})
                
                # Extract face embeddings
                try:
                    face_embedding = extract_face_embeddings(filepath)
                    request.user.set_face_embedding(face_embedding)
                    request.user.save()
                    
                    messages.success(request, 'Face registration successful!')
                    return redirect('profile')
                except Exception as e:
                    print(f"Face registration error: {str(e)}")
                    messages.error(request, 'Could not detect a face. Please try again.')
                    os.remove(filepath)
                    return render(request, 'detector/register_face.html', {'form': form})
            
            except Exception as e:
                print(f"Error during face registration: {str(e)}")
                messages.error(request, 'An error occurred. Please try again.')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return render(request, 'detector/register_face.html', {'form': form})
    else:
        form = FaceRegistrationForm()
    
    return render(request, 'detector/register_face.html', {'form': form})

@login_required
def delete_analysis(request, analysis_id):
    analysis = get_object_or_404(AnalysisHistory, id=analysis_id, user=request.user)
    if request.method == 'POST':
        # Delete the associated image file
        if analysis.image:
            file_path = os.path.join(settings.MEDIA_ROOT, str(analysis.image))
            if os.path.exists(file_path):
                os.remove(file_path)
        analysis.delete()
        messages.success(request, 'Analysis deleted successfully!')
    return redirect('analyze')

def about(request):
    return render(request, 'detector/about.html')