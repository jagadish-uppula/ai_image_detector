from django.urls import path
from . import views
from .views import delete_analysis, update_profile_pic

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('login/', views.user_login, name='login'),
    path('register/', views.user_register, name='register'),
    path('logout/', views.user_logout, name='logout'),
    path('analyze/', views.analyze, name='analyze'),
    path('predict/', views.predict, name='predict'),
    path('visualize/', views.visualize, name='visualize'),
    path('security/', views.security, name='security'),
    path('register-face/', views.register_face, name='register_face'),
    path('about/', views.about, name='about'),
    path('profile/', views.profile, name='profile'),
    path('update-profile-pic/', update_profile_pic, name='update_profile_pic'),
    path('delete-analysis/<int:analysis_id>/', delete_analysis, name='delete-analysis'),
    path('verify-face/', views.verify_face, name='verify_face'),
]