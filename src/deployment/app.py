"""
Flask backend for MuseAI web application.
"""

import os
import sys
from pathlib import Path
import uuid
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import torch

sys.path.append(str(Path(__file__).parent.parent))
from deployment.model_inference import StyleTransferInference

app = Flask(__name__)

# Configuration - use absolute paths from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
UPLOAD_FOLDER = str(PROJECT_ROOT / 'outputs' / 'uploads')
RESULT_FOLDER = str(PROJECT_ROOT / 'outputs' / 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Global model instance
model = None

# Store mapping of file_id to original filename
filename_map = {}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the trained model"""
    global model
    
    checkpoint_path = Path(__file__).parent.parent.parent / 'checkpoints' / 'final_model.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
        print("   Please train the model first using: python src/training/train.py")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("LOADING MUSEAI MODEL")
    print("="*70)
    
    model = StyleTransferInference(checkpoint_path=str(checkpoint_path))
    
    print("✅ Model loaded successfully")
    print("="*70 + "\n")


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/stylize', methods=['POST'])
def stylize():
    """
    API endpoint for stylizing images.
    
    Expects:
        - file: Image file (face photo)
        - artist: Artist name ('picasso' or 'rembrandt')
        - alpha: (optional) Style strength (default 1.0)
    
    Returns:
        JSON with result image URL
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png'}), 400
        
        # Get artist
        artist = request.form.get('artist', 'picasso')
        if artist not in ['picasso', 'rembrandt']:
            return jsonify({'error': 'Invalid artist. Choose picasso or rembrandt'}), 400
        
        # Get alpha (style strength)
        try:
            alpha = float(request.form.get('alpha', 1.0))
            alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
        except:
            alpha = 1.0
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        original_name_no_ext = original_filename.rsplit('.', 1)[0]
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}.{file_ext}')
        file.save(upload_path)
        
        # Store original filename mapping
        filename_map[file_id] = original_filename
        
        print(f"📸 Received image: {original_filename}")
        print(f"   Artist: {artist}")
        print(f"   Style strength: {alpha}")
        
        # Stylize
        print("🎨 Stylizing...")
        stylized = model.stylize_with_random_style(upload_path, artist, alpha=alpha)
        
        # Save result
        result_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}.jpg')
        stylized.save(result_path, quality=95)
        
        print(f"✅ Stylization complete")
        
        # Return result URL with original filename
        return jsonify({
            'success': True,
            'result_url': f'/api/result/{file_id}.jpg',
            'download_url': f'/api/download/{file_id}',
            'artist': artist,
            'original_filename': original_filename
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/result/<filename>')
def get_result(filename):
    """Serve result image for preview"""
    try:
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        if not os.path.exists(result_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(result_path, mimetype='image/jpeg')
    except Exception as e:
        print(f"Error serving result: {e}")
        return jsonify({'error': 'File not found'}), 404


@app.route('/api/download/<file_id>')
def download_result(file_id):
    """Download result image with original filename"""
    try:
        result_path = os.path.join(app.config['RESULT_FOLDER'], f'{file_id}.jpg')
        
        if not os.path.exists(result_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Get original filename
        original_filename = filename_map.get(file_id, 'result.jpg')
        
        # Return file with original name
        return send_file(
            result_path,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=original_filename
        )
    except Exception as e:
        print(f"Error downloading result: {e}")
        return jsonify({'error': 'Download failed'}), 500


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    print("\n" + "="*70)
    print("STARTING MUSEAI WEB APP")
    print("="*70)
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    print("\n📸 Upload a face photo and choose an artist to stylize!")
    print("\nPress Ctrl+C to stop the server\n")
    print("="*70 + "\n")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)