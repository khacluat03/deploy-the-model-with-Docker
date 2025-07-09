from flask import Flask, request, render_template, send_file, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename
import torch
from ultralytics import YOLO
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load your YOLO model
# Replace 'your_model.pt' with your actual model path
MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def process_image(image_path):
    """Process image with YOLO model and create visualization"""
    # Run YOLO inference
    results = model(image_path)

    # Load original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create matplotlib figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.axis('off')

    # Draw bounding boxes and labels
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Get confidence and class
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                # Get class name
                class_name = model.names[cls]

                # Draw bounding box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

                # Add label
                ax.text(
                    x1, y1 - 10,
                    f'{class_name}: {conf:.2f}',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.8),
                    fontsize=10, color='white'
                )

    # Save plot to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    plt.close()

    return img_buffer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Process image
            result_buffer = process_image(filepath)

            # Save result
            result_filename = f"result_{filename.rsplit('.', 1)[0]}.png"
            result_path = os.path.join(RESULT_FOLDER, result_filename)

            with open(result_path, 'wb') as f:
                f.write(result_buffer.getvalue())

            # Convert to base64 for display
            result_buffer.seek(0)
            img_base64 = base64.b64encode(result_buffer.read()).decode()

            # Clean up uploaded file
            os.remove(filepath)

            return jsonify({
                'success': True,
                'image': img_base64,
                'filename': result_filename
            })

        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
