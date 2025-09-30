from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
import pickle
import numpy as np
import os
from django.conf import settings

def validate_image_size(value):
    """Validate that uploaded image size doesn't exceed 5MB"""
    filesize = value.size
    if filesize > 5 * 1024 * 1024:  # 5MB limit
        raise ValidationError("The maximum file size that can be uploaded is 5MB")

def user_upload_path(instance, filename):
    """Generate upload path for user-uploaded files"""
    return os.path.join('uploads', str(instance.user.id), filename)

class User(AbstractUser):
    face_embedding = models.BinaryField(null=True, blank=True)
    profile_picture = models.ImageField(
        upload_to='profile_pics/',
        validators=[validate_image_size],
        null=True,
        blank=True
    )
    is_face_registered = models.BooleanField(default=False)

    def set_face_embedding(self, embedding):
        """Store face embedding as binary"""
        if embedding is not None:
            self.face_embedding = pickle.dumps(embedding)
            self.is_face_registered = True
            return True
        return False

    def get_face_embedding(self):
        """Retrieve face embedding as numpy array"""
        if self.face_embedding:
            return pickle.loads(self.face_embedding)
        return None

    # Remove or modify the clean() method to be less strict
    def clean(self):
        """Only validate if we're not in the registration process"""
        super().clean()
        if self.pk and self.profile_picture and not self.face_embedding:
            raise ValidationError("Face embedding must be set when profile picture is provided")

    # Remove the save() override that was trying to extract embeddings
    # We'll handle this in the view instead

    def __str__(self):
        return self.username

class AnalysisHistory(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    image = models.ImageField(upload_to=user_upload_path)
    is_ai_generated = models.BooleanField()
    confidence = models.FloatField()
    similarity_score = models.FloatField(null=True, blank=True)
    face_embedding = models.BinaryField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {'AI' if self.is_ai_generated else 'Real'} - {self.created_at}"

    def get_face_embedding(self):
        """Retrieve face embedding from analysis"""
        if self.face_embedding:
            return pickle.loads(self.face_embedding)
        return None

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Analysis Histories'