from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import check_password
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth.backends import ModelBackend
from .db_utils import get_db

User = get_user_model()

class EmailOrUsernameModelBackend(ModelBackend):
    """
    Authenticate against either username or email
    """
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            # Try to fetch user by username
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            try:
                # Try to fetch user by email
                user = User.objects.get(email=username)
            except User.DoesNotExist:
                return None
        
        if user.check_password(password):
            return user
        return None

def verify_login(username, password):
    """
    Verify user credentials and return user object if valid
    """
    try:
        # First try username
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        try:
            # Then try email
            user = User.objects.get(email=username)
        except User.DoesNotExist:
            return None
    
    if user.check_password(password):
        return user
    return None

def load_user(user_id):
    """
    Load user by ID (required for Flask-Login compatibility)
    """
    try:
        return User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return None

def get_user_by_email(email):
    """
    Get user by email address
    """
    try:
        return User.objects.get(email=email)
    except User.DoesNotExist:
        return None

def validate_password(user, password):
    """
    Validate user's password
    """
    return check_password(password, user.password)

def update_user_face_embedding(user, embedding):
    """
    Update user's face embedding
    """
    try:
        user.face_embedding = embedding
        user.save()
        return True
    except Exception as e:
        print(f"Error updating face embedding: {e}")
        return False

def create_user(username, email, password, full_name=None, is_admin=False):
    """
    Create a new user with the given credentials
    """
    try:
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=full_name.split()[0] if full_name else '',
            last_name=' '.join(full_name.split()[1:]) if full_name else ''
        )
        user.is_admin = is_admin
        user.save()
        return user
    except Exception as e:
        print(f"Error creating user: {e}")
        return None