from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User
from django.core.validators import FileExtensionValidator

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control'}))
    first_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={'class': 'form-control'}))
    last_name = forms.CharField(max_length=30, required=True, widget=forms.TextInput(attrs={'class': 'form-control'}))
    profile_picture = forms.ImageField(
        required=True,
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png'])],
        help_text='Upload a clear photo of your face (JPEG or PNG)',
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'})
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 
                 'profile_picture', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs.update({'class': 'form-control'})

    def clean_profile_picture(self):
        profile_pic = self.cleaned_data.get('profile_picture')
        if profile_pic:
            if profile_pic.size > 5 * 1024 * 1024:  # 5MB limit
                raise ValidationError("The maximum file size that can be uploaded is 5MB")
        return profile_pic


class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)

class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label='Upload Image',
        help_text='Upload an image to analyze'
    )

class FaceVerificationForm(forms.Form):
    verification_image = forms.ImageField(
        required=True,
        help_text='Upload or capture an image for verification'
    )

class FaceRegistrationForm(forms.Form):
    face_image = forms.ImageField(
        required=True,
        help_text='Upload or capture a clear photo of your face'
    )