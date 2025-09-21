#!/usr/bin/env python3
"""
OMR Evaluation Web Backend API
Flask-based REST API for OMR evaluation system
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
from pathlib import Path
from datetime import datetime
import logging
from werkzeug.utils import secure_filename

# Import our modules
from database import OMRDatabase
from omr_processor import OMRProcessor
from enhanced_omr_processor import EnhancedOMRProcessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
db = OMRDatabase()

# Global processors (will be loaded when needed)
processor = None
enhanced_processor = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_processor():
    """Load the OMR processor with trained model"""
    global processor
    if processor is None:
        model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
        if Path(model_path).exists():
            try:
                processor = OMRProcessor(model_path)
                logging.info("OMR processor loaded successfully")
            except Exception as e:
                logging.error(f"Error loading processor: {e}")
                return False
        else:
            logging.error(f"Model not found: {model_path}")
            return False
    return True

def load_enhanced_processor():
    """Load the enhanced OMR processor with trained model"""
    global enhanced_processor
    if enhanced_processor is None:
        model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
        if Path(model_path).exists():
            try:
                enhanced_processor = EnhancedOMRProcessor(model_path)
                logging.info("Enhanced OMR processor loaded successfully")
            except Exception as e:
                logging.error(f"Error loading enhanced processor: {e}")
                return False
        else:
            logging.error(f"Model not found: {model_path}")
            return False
    return True

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': processor is not None
    })

@app.route('/api/students', methods=['GET'])
def get_students():
    """Get all students"""
    try:
        with db.db_path as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students ORDER BY created_at DESC")
            students = cursor.fetchall()
            
            result = []
            for student in students:
                result.append({
                    'id': student[0],
                    'student_id': student[1],
                    'name': student[2],
                    'email': student[3],
                    'created_at': student[4]
                })
            
            return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students', methods=['POST'])
def add_student():
    """Add a new student"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        name = data.get('name')
        email = data.get('email')
        
        if not student_id or not name:
            return jsonify({'error': 'student_id and name are required'}), 400
        
        db_id = db.add_student(student_id, name, email)
        
        return jsonify({
            'id': db_id,
            'student_id': student_id,
            'name': name,
            'email': email,
            'message': 'Student added successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exams', methods=['GET'])
def get_exams():
    """Get all exams"""
    try:
        with db.db_path as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM exams ORDER BY created_at DESC")
            exams = cursor.fetchall()
            
            result = []
            for exam in exams:
                result.append({
                    'id': exam[0],
                    'exam_name': exam[1],
                    'exam_date': exam[2],
                    'total_questions': exam[3],
                    'subjects': json.loads(exam[4]),
                    'answer_key': json.loads(exam[5]),
                    'created_at': exam[6]
                })
            
            return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/exams', methods=['POST'])
def add_exam():
    """Add a new exam"""
    try:
        data = request.get_json()
        exam_name = data.get('exam_name')
        exam_date = data.get('exam_date')
        total_questions = data.get('total_questions')
        subjects = data.get('subjects', [])
        answer_key = data.get('answer_key', {})
        
        if not all([exam_name, exam_date, total_questions]):
            return jsonify({'error': 'exam_name, exam_date, and total_questions are required'}), 400
        
        exam_id = db.add_exam(exam_name, exam_date, total_questions, subjects, answer_key)
        
        return jsonify({
            'id': exam_id,
            'exam_name': exam_name,
            'exam_date': exam_date,
            'total_questions': total_questions,
            'subjects': subjects,
            'answer_key': answer_key,
            'message': 'Exam added successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_omr_sheet():
    """Upload and process OMR sheet"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get additional parameters
        student_id = request.form.get('student_id')
        exam_id = request.form.get('exam_id')
        sheet_version = request.form.get('sheet_version', 'Set A')
        
        if not student_id or not exam_id:
            return jsonify({'error': 'student_id and exam_id are required'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Load processor if not already loaded
        if not load_processor():
            return jsonify({'error': 'Model not available'}), 500
        
        # Get exam answer key
        with db.db_path as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT answer_key FROM exams WHERE id = ?", (exam_id,))
            result = cursor.fetchone()
            if not result:
                return jsonify({'error': 'Exam not found'}), 404
            answer_key = json.loads(result[0])
        
        # Process OMR sheet
        result = processor.process_omr_sheet(filepath, answer_key, sheet_version)
        
        # Save to database
        omr_sheet_id = db.add_omr_sheet(student_id, exam_id, filepath, sheet_version)
        db.update_omr_processing(omr_sheet_id, filepath, 'completed')
        
        # Save evaluation result
        evaluation_id = db.add_evaluation_result(
            omr_sheet_id,
            result['subject_scores'],
            result['total_score'],
            result['percentage'],
            result['student_answers'],
            result['processing_time'],
            result['model_confidence']
        )
        
        # Add audit entry
        db.add_audit_entry(
            omr_sheet_id,
            'omr_processed',
            {
                'evaluation_id': evaluation_id,
                'processing_time': result['processing_time'],
                'model_confidence': result['model_confidence']
            }
        )
        
        return jsonify({
            'success': True,
            'omr_sheet_id': omr_sheet_id,
            'evaluation_id': evaluation_id,
            'result': result,
            'message': 'OMR sheet processed successfully'
        })
    
    except Exception as e:
        logging.error(f"Error processing OMR sheet: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/enhanced', methods=['POST'])
def upload_omr_sheet_enhanced():
    """Upload and process OMR sheet with enhanced detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Get additional parameters
        student_id = request.form.get('student_id')
        exam_id = request.form.get('exam_id')
        sheet_version = request.form.get('sheet_version', 'Set A')
        
        if not student_id or not exam_id:
            return jsonify({'error': 'student_id and exam_id are required'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Load enhanced processor if not already loaded
        if not load_enhanced_processor():
            return jsonify({'error': 'Enhanced model not available'}), 500
        
        # Get exam answer key
        with db.db_path as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT answer_key FROM exams WHERE id = ?", (exam_id,))
            result = cursor.fetchone()
            if not result:
                return jsonify({'error': 'Exam not found'}), 404
            answer_key = json.loads(result[0])
        
        # Process OMR sheet with enhanced detection
        result = enhanced_processor.process_and_save(
            filepath, 
            int(student_id), 
            int(exam_id), 
            answer_key, 
            sheet_version
        )
        
        if result.get('success', True):  # Default to True if not specified
            return jsonify({
                'success': True,
                'result': result,
                'message': 'OMR sheet processed successfully with enhanced detection'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'message': 'Failed to process OMR sheet'
            }), 500
    
    except Exception as e:
        logging.error(f"Error processing OMR sheet with enhanced detection: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<int:student_id>', methods=['GET'])
def get_student_results(student_id):
    """Get results for a specific student"""
    try:
        exam_id = request.args.get('exam_id', type=int)
        results = db.get_student_results(str(student_id), exam_id)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_all_results():
    """Get all evaluation results"""
    try:
        limit = request.args.get('limit', 50, type=int)
        results = db.get_recent_evaluations(limit)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics/<int:exam_id>', methods=['GET'])
def get_exam_statistics(exam_id):
    """Get statistics for a specific exam"""
    try:
        stats = db.get_exam_statistics(exam_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/processing/details/<int:omr_sheet_id>', methods=['GET'])
def get_processing_details(omr_sheet_id):
    """Get detailed processing information for an OMR sheet"""
    try:
        with db.db_path as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get OMR sheet details
            cursor.execute("""
                SELECT os.*, s.student_id, s.name, e.exam_name, er.*
                FROM omr_sheets os
                JOIN students s ON os.student_id = s.id
                JOIN exams e ON os.exam_id = e.id
                LEFT JOIN evaluation_results er ON er.omr_sheet_id = os.id
                WHERE os.id = ?
            """, (omr_sheet_id,))
            
            result = cursor.fetchone()
            if not result:
                return jsonify({'error': 'OMR sheet not found'}), 404
            
            # Get audit trail
            cursor.execute("""
                SELECT * FROM audit_trail 
                WHERE omr_sheet_id = ? 
                ORDER BY timestamp DESC
            """, (omr_sheet_id,))
            
            audit_entries = []
            for row in cursor.fetchall():
                audit_entries.append({
                    'action': row['action'],
                    'details': json.loads(row['details']) if row['details'] else None,
                    'timestamp': row['timestamp'],
                    'user_id': row['user_id']
                })
            
            return jsonify({
                'omr_sheet': dict(result),
                'audit_trail': audit_entries
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/preview/<int:omr_sheet_id>', methods=['GET'])
def get_detection_preview(omr_sheet_id):
    """Get detection preview with question numbers and bubbles mapped"""
    try:
        with db.db_path as conn:
            cursor = conn.cursor()
            
            # Get OMR sheet image path
            cursor.execute("SELECT sheet_image_path FROM omr_sheets WHERE id = ?", (omr_sheet_id,))
            result = cursor.fetchone()
            if not result:
                return jsonify({'error': 'OMR sheet not found'}), 404
            
            image_path = result[0]
            
            # Load enhanced processor if not already loaded
            if not load_enhanced_processor():
                return jsonify({'error': 'Enhanced model not available'}), 500
            
            # Get exam answer key
            cursor.execute("""
                SELECT e.answer_key FROM exams e
                JOIN omr_sheets os ON e.id = os.exam_id
                WHERE os.id = ?
            """, (omr_sheet_id,))
            result = cursor.fetchone()
            if not result:
                return jsonify({'error': 'Exam not found'}), 404
            
            answer_key = json.loads(result[0])
            
            # Process image for detection preview
            processed_image = enhanced_processor.preprocess_image(image_path)
            question_numbers = enhanced_processor.detect_question_numbers(processed_image)
            bubbles = enhanced_processor.detect_bubbles_enhanced(processed_image)
            question_mapping = enhanced_processor.map_bubbles_to_questions(question_numbers, bubbles)
            
            return jsonify({
                'question_numbers': question_numbers,
                'bubbles': bubbles,
                'question_mapping': question_mapping,
                'image_path': image_path
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<int:exam_id>', methods=['GET'])
def export_exam_results(exam_id):
    """Export exam results to CSV"""
    try:
        output_path = f"exports/exam_{exam_id}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.makedirs('exports', exist_ok=True)
        
        db.export_results_csv(exam_id, output_path)
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"exam_{exam_id}_results.csv",
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/performance', methods=['POST'])
def add_model_performance():
    """Add model performance metrics"""
    try:
        data = request.get_json()
        
        required_fields = ['model_name', 'model_version', 'accuracy', 'precision', 'recall', 'f1_score', 'model_path']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        db.add_model_performance(
            data['model_name'],
            data['model_version'],
            data['accuracy'],
            data['precision'],
            data['recall'],
            data['f1_score'],
            data['model_path'],
            data.get('training_date')
        )
        
        return jsonify({'message': 'Model performance added successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch/upload', methods=['POST'])
def batch_upload():
    """Process multiple OMR sheets in batch"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        exam_id = request.form.get('exam_id')
        sheet_version = request.form.get('sheet_version', 'Set A')
        
        if not exam_id:
            return jsonify({'error': 'exam_id is required'}), 400
        
        if not load_processor():
            return jsonify({'error': 'Model not available'}), 500
        
        # Get exam answer key
        with db.db_path as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT answer_key FROM exams WHERE id = ?", (exam_id,))
            result = cursor.fetchone()
            if not result:
                return jsonify({'error': 'Exam not found'}), 404
            answer_key = json.loads(result[0])
        
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Save file
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{timestamp}_{filename}"
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    file.save(filepath)
                    
                    # Process sheet
                    result = processor.process_omr_sheet(filepath, answer_key, sheet_version)
                    result['filename'] = file.filename
                    results.append(result)
                    
                except Exception as e:
                    results.append({
                        'filename': file.filename,
                        'error': str(e)
                    })
        
        return jsonify({
            'success': True,
            'processed_files': len([r for r in results if 'error' not in r]),
            'failed_files': len([r for r in results if 'error' in r]),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('exports', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)