// Initialize tooltips
$(function () {
    $('[data-bs-toggle="tooltip"]').tooltip()
})

// Initialize popovers
$(function () {
    $('[data-bs-toggle="popover"]').popover()
})

// Handle file upload previews
document.addEventListener('DOMContentLoaded', function() {
    // Profile picture preview
    const profilePicInput = document.getElementById('id_profile_picture');
    if (profilePicInput) {
        profilePicInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    const preview = document.getElementById('profilePicPreview');
                    if (preview) {
                        preview.src = event.target.result;
                        preview.style.display = 'block';
                        document.querySelector('.profile-pic-placeholder').style.display = 'none';
                    }
                }
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Image preview for analysis
    const imageInput = document.getElementById('id_image');
    if (imageInput) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    const preview = document.getElementById('imagePreview');
                    if (preview) {
                        preview.src = event.target.result;
                        preview.style.display = 'block';
                        document.querySelector('.upload-area').style.display = 'none';
                    }
                }
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Handle camera capture
    const openCameraBtn = document.getElementById('openCameraBtn');
    if (openCameraBtn) {
        openCameraBtn.addEventListener('click', function() {
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function(stream) {
                        const preview = document.getElementById('cameraPreview');
                        preview.srcObject = stream;
                        document.getElementById('captureBtn').style.display = 'block';
                    })
                    .catch(function(error) {
                        alert('Could not access the camera: ' + error.message);
                    });
            } else {
                alert('Camera access is not supported by your browser');
            }
        });
    }
    
    // Handle face capture
    const captureBtn = document.getElementById('captureBtn');
    if (captureBtn) {
        captureBtn.addEventListener('click', function() {
            const video = document.getElementById('cameraPreview');
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const imageData = canvas.toDataURL('image/jpeg');
            const preview = document.getElementById('imagePreview') || document.getElementById('verificationPreview');
            preview.src = imageData;
            preview.style.display = 'block';
            
            // Stop camera stream
            video.srcObject.getVideoTracks().forEach(track => track.stop());
            $('#cameraModal').modal('hide');
            
            // Create a blob from the canvas and set it as the file input
            canvas.toBlob(function(blob) {
                const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' });
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                const fileInput = document.getElementById('id_image') || document.getElementById('id_verification_image');
                fileInput.files = dataTransfer.files;
            }, 'image/jpeg');
        });
    }
});

// Handle AJAX form submissions
$(document).on('submit', '.ajax-form', function(e) {
    e.preventDefault();
    const form = $(this);
    const url = form.attr('action');
    const method = form.attr('method');
    const data = form.serialize();
    
    $.ajax({
        url: url,
        type: method,
        data: data,
        success: function(response) {
            if (response.success) {
                // Handle success
                if (response.redirect) {
                    window.location.href = response.redirect;
                }
            } else {
                // Handle errors
                alert(response.message || 'An error occurred');
            }
        },
        error: function(xhr) {
            alert('An error occurred: ' + xhr.statusText);
        }
    });
});

// Handle messages timeout
setTimeout(function() {
    $('.alert').fadeOut('slow');
}, 5000);