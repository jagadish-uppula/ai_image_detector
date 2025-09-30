from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, AnalysisHistory

class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'is_staff', 'get_face_registered')  # Removed get_is_admin
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'is_face_registered')
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'email', 'profile_picture')}),
        ('Permissions', {
            'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions'),
        }),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
        ('Face Verification', {'fields': ('is_face_registered',)}),
    )
    readonly_fields = ('is_face_registered',)
    filter_horizontal = ('groups', 'user_permissions',)

    def get_face_registered(self, obj):
        return obj.is_face_registered
    get_face_registered.short_description = 'Face Registered'
    get_face_registered.boolean = True

@admin.register(AnalysisHistory)
class AnalysisHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'is_ai_generated', 'confidence', 'similarity_score', 'created_at')
    list_filter = ('is_ai_generated', 'created_at')
    search_fields = ('user__username',)
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'

admin.site.register(User, CustomUserAdmin)