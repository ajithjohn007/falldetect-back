from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = YOLO('yolo11s-pose.pt')

@app.route('/detect_fall', methods=['POST'])
def detect_fall():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded video
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(filepath)
    
    try:
        # Process video with YOLO model
        results = model(filepath)
        # Process results as needed
        # This is a placeholder - modify according to your actual processing logic
        processed_results = {"status": "success", "message": "Fall detection completed"}
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(processed_results)
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080) 