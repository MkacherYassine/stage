from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def resize_image(image, width):
    """Resize the image to the given width while maintaining the aspect ratio."""
    aspect_ratio = width / float(image.shape[1])
    dimensions = (width, int(image.shape[0] * aspect_ratio))
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

def compress_image(image_path, max_size_kb):
    """Compress the image to ensure its size does not exceed max_size_kb."""
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Unable to load image."

    quality = 95  # Start with high quality
    while True:
        is_success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        size_kb = len(buffer) / 1024  # Size in KB
        if size_kb <= max_size_kb or quality <= 10:
            with open(image_path, "wb") as f:
                f.write(buffer)
            break
        quality -= 5

def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Unable to load image.", None, 0
    
    # Save the original uploaded image with a unique name
    original_filename = str(uuid.uuid4()) + '_original.jpg'
    original_path = os.path.join(UPLOAD_FOLDER, original_filename)
    cv2.imwrite(original_path, image)
    
    # Resize the image to a width of 1200 pixels while maintaining the aspect ratio
    resized_image = resize_image(image, 1200)
    
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) == 0:
        return "No faces detected", None, 0
    
    # Assuming only one face is needed
    x, y, w, h = faces[0]
    
    # Make the rectangle slightly bigger
    padding = 40
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(resized_image.shape[1] - x, w + 2 * padding)
    h = min(resized_image.shape[0] - y, h + 2 * padding)
    
    # Crop the face
    face_image = resized_image[y:y+h, x:x+w]
    
    # Resize the face image to the real-life ID photo dimensions (132 x 170 pixels)
    id_photo_dimensions = (132, 170)
    face_image_resized = cv2.resize(face_image, id_photo_dimensions)
    
    # Save the cropped and resized face image with a unique name
    processed_filename = str(uuid.uuid4()) + '_processed.jpg'
    processed_path = os.path.join(UPLOAD_FOLDER, processed_filename)
    
    # Save the resized image first
    cv2.imwrite(processed_path, face_image_resized)
    
    # Compress the image to ensure it's not larger than 1200 KB
    compress_image(processed_path, max_size_kb=1200)
    
    return processed_filename, original_path, len(faces)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            output_image, original_path, face_count = detect_faces(file_path)
            if face_count != 1:
                return render_template('index.html', error="Please upload an image with exactly one face or make sure it's a human image.")
            else:
                # Delete the original uploaded image after displaying the result
                os.remove(original_path)
                return render_template('result.html', face_count=face_count, output_image=output_image)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
