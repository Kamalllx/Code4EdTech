"""
Flask Application for OMR Evaluation System
Real-time camera capture, preprocessing, and YOLO-based bubble detection
"""

import os
import cv2
import json
import base64
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import uuid

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import ssl

# Import our custom modules
from services.camera_service import CameraService
from services.preprocessing_service import PreprocessingService
from services.yolo_service import YOLOService
from services.audit_service import AuditService
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize services
camera_service = CameraService()
preprocessing_service = PreprocessingService()
yolo_service = YOLOService(app.config['MODEL_PATH'])
audit_service = AuditService(app.config['AUDIT_DIR'])

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """Main dashboard for OMR evaluation system"""
    return render_template('index.html')


@app.route('/camera')
def camera_page():
    """Camera capture page"""
    return render_template('camera.html')


@app.route('/api/camera/status')
def camera_status():
    """Check camera availability and status"""
    try:
        status = camera_service.get_camera_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/camera/capture', methods=['POST'])
def capture_image():
    """Capture image from camera (PC/Phone/AR)"""
    try:
        data = request.json
        camera_type = data.get('camera_type', 'pc')  # pc, phone, ar
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Capture image based on camera type
        if camera_type == 'upload':
            # Handle file upload
            if 'image_data' not in data:
                return jsonify({
                    'success': False,
                    'error': 'No image data provided'
                }), 400
            
            # Decode base64 image
            image_data = data['image_data']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            np_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
        else:
            # Capture from camera
            image = camera_service.capture_image(camera_type)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to capture image'
            }), 500
        
        # Save original image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"capture_{session_id}_{timestamp}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(image_path, image)
        
        # Create audit entry
        audit_entry = audit_service.create_capture_entry(
            session_id, camera_type, image_path, image.shape
        )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'image_url': url_for('uploaded_file', filename=filename),
            'image_info': {
                'width': image.shape[1],
                'height': image.shape[0],
                'channels': image.shape[2] if len(image.shape) > 2 else 1
            },
            'audit_id': audit_entry['audit_id']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess_image():
    """Preprocess captured image for OMR evaluation"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID required'
            }), 400
        
        # Find the image file
        upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                       if f.startswith(f"capture_{session_id}")]
        
        if not upload_files:
            return jsonify({
                'success': False,
                'error': 'Original image not found'
            }), 404
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_files[0])
        image = cv2.imread(image_path)
        
        # Preprocess image
        preprocessing_config = data.get('config', {})
        processed_image, preprocessing_info = preprocessing_service.preprocess_omr_image(
            image, preprocessing_config
        )
        
        # Save preprocessed image
        processed_filename = f"processed_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        processed_path = os.path.join(app.config['RESULTS_FOLDER'], processed_filename)
        cv2.imwrite(processed_path, processed_image)
        
        # Update audit trail
        audit_service.update_preprocessing(session_id, preprocessing_info, processed_path)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'processed_image_url': url_for('result_file', filename=processed_filename),
            'preprocessing_info': preprocessing_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_omr():
    """Evaluate OMR sheet using YOLO model"""
    try:
        data = request.json
        session_id = data.get('session_id')
        answer_key = data.get('answer_key', {})
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID required'
            }), 400
        
        # Find the processed image
        result_files = [f for f in os.listdir(app.config['RESULTS_FOLDER']) 
                       if f.startswith(f"processed_{session_id}")]
        
        if not result_files:
            return jsonify({
                'success': False,
                'error': 'Processed image not found'
            }), 404
        
        processed_path = os.path.join(app.config['RESULTS_FOLDER'], result_files[0])
        image = cv2.imread(processed_path)
        
        # Run YOLO evaluation
        evaluation_results = yolo_service.evaluate_omr_sheet(image, answer_key)
        
        # Generate evaluation report
        report_filename = f"evaluation_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create annotated image with results
        annotated_image = yolo_service.create_annotated_image(image, evaluation_results)
        annotated_filename = f"annotated_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        annotated_path = os.path.join(app.config['RESULTS_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, annotated_image)
        
        # Update audit trail
        audit_service.update_evaluation(session_id, evaluation_results, report_path)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'evaluation_results': evaluation_results,
            'annotated_image_url': url_for('result_file', filename=annotated_filename),
            'report_url': url_for('result_file', filename=report_filename)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch_evaluate', methods=['POST'])
def batch_evaluate():
    """Batch evaluate multiple OMR sheets"""
    try:
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        files = request.files.getlist('images')
        answer_key = json.loads(request.form.get('answer_key', '{}'))
        
        batch_id = str(uuid.uuid4())
        results = []
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            saved_filename = f"batch_{batch_id}_{i}_{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(file_path)
            
            # Process image
            image = cv2.imread(file_path)
            processed_image, preprocessing_info = preprocessing_service.preprocess_omr_image(image)
            
            # Evaluate with YOLO
            evaluation_results = yolo_service.evaluate_omr_sheet(processed_image, answer_key)
            
            results.append({
                'filename': filename,
                'evaluation_results': evaluation_results,
                'preprocessing_info': preprocessing_info
            })
        
        # Save batch results
        batch_report_filename = f"batch_report_{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        batch_report_path = os.path.join(app.config['RESULTS_FOLDER'], batch_report_filename)
        
        batch_summary = {
            'batch_id': batch_id,
            'total_sheets': len(results),
            'processed_at': datetime.now().isoformat(),
            'results': results
        }
        
        with open(batch_report_path, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'results': results,
            'report_url': url_for('result_file', filename=batch_report_filename)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/audit/<session_id>')
def get_audit_trail(session_id):
    """Get audit trail for a specific session"""
    try:
        audit_data = audit_service.get_session_audit(session_id)
        if not audit_data:
            return jsonify({
                'success': False,
                'error': 'Audit trail not found'
            }), 404
        
        return jsonify({
            'success': True,
            'audit_data': audit_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check all services
        camera_status = camera_service.get_camera_status()
        model_status = yolo_service.check_model_status()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'services': {
                'camera': camera_status,
                'yolo_model': model_status,
                'preprocessing': 'active',
                'audit': 'active'
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))


@app.route('/dashboard')
def dashboard():
    """Administrative dashboard"""
    try:
        # Get system statistics
        stats = audit_service.get_system_statistics()
        return render_template('dashboard.html', stats=stats)
    except Exception as e:
        return render_template('dashboard.html', error=str(e))


if __name__ == '__main__':
    # Create SSL context for HTTPS (required for camera access)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # For development, create self-signed certificate
    if not os.path.exists('cert.pem') or not os.path.exists('key.pem'):
        print("Creating self-signed certificate for HTTPS...")
        os.system('openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"')
    
    try:
        context.load_cert_chain('cert.pem', 'key.pem')
        print("Starting OMR Flask Application with HTTPS...")
        print("Access the application at: https://localhost:5000")
        print("Note: You may need to accept the self-signed certificate warning")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=app.config['DEBUG'],
            ssl_context=context
        )
    except Exception as e:
        print(f"Failed to start with HTTPS: {e}")
        print("Starting with HTTP (camera access may be limited)...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=app.config['DEBUG']
        )