# OMR Flask Application - Complete Documentation

## 🎯 Project Overview

This is a comprehensive **Optical Mark Recognition (OMR) Flask Application** that provides intelligent answer sheet evaluation using computer vision and AI technologies. The system supports multi-device camera capture, advanced image preprocessing, YOLO-based bubble detection, and automated scoring with complete audit trails.

### ✨ Key Features

- **Multi-Device Camera Support**: PC webcam, phone camera, and AR device compatibility
- **HTTPS/SSL Security**: Secure camera access with self-signed certificates
- **Advanced Preprocessing**: Noise reduction, perspective correction, and OMR-specific optimizations  
- **YOLO AI Detection**: YOLOv8-powered bubble detection with computer vision fallback
- **Automated Scoring**: Instant evaluation with confidence metrics and detailed analytics
- **Batch Processing**: Handle multiple answer sheets simultaneously
- **Complete Audit Trail**: Comprehensive logging and compliance tracking
- **Modern Web Interface**: Responsive UI with real-time dashboards

## 🏗️ Architecture Overview

```
omr_flask_app/
├── app.py                 # Main Flask application
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── services/             # Business logic layer
│   ├── camera_service.py     # Camera capture handling
│   ├── preprocessing_service.py  # Image enhancement
│   ├── yolo_service.py       # AI model integration
│   └── audit_service.py      # Audit and compliance
├── templates/            # HTML templates
│   ├── index.html           # Homepage
│   ├── camera.html          # Camera interface
│   └── dashboard.html       # Analytics dashboard
├── uploads/              # File storage
├── processed/            # Processed images
├── reports/              # Generated reports
└── audit_logs/           # Audit trail storage
```

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.8+**
- **OpenCV 4.5+**
- **PyTorch** (for YOLO)
- **Modern web browser** with camera support

### Installation

1. **Clone and Setup**
   ```bash
   cd draft2/Code4EdTech/omr_flask_app
   pip install -r requirements.txt
   ```

2. **Install Required Dependencies**
   ```bash
   pip install flask flask-cors opencv-python ultralytics torch torchvision pillow numpy scikit-image
   ```

3. **Setup Directory Structure**
   ```bash
   mkdir -p uploads processed reports audit_logs ssl_certs
   ```

4. **Generate SSL Certificates** (for HTTPS camera access)
   ```bash
   # Windows PowerShell
   openssl req -x509 -newkey rsa:4096 -keyout ssl_certs/key.pem -out ssl_certs/cert.pem -days 365 -nodes
   ```

5. **Download YOLO Model** (if not present)
   ```bash
   # The app will automatically download yolov8n-cls.pt on first run
   ```

### Running the Application

1. **Development Mode**
   ```bash
   python app.py
   ```
   - Runs on `https://localhost:5000` (HTTPS for camera access)
   - Auto-reloads on code changes
   - Detailed debugging information

2. **Production Mode**
   ```bash
   # Set environment variable
   set FLASK_ENV=production  # Windows
   export FLASK_ENV=production  # Linux/Mac
   
   python app.py
   ```

## 📡 API Documentation

### Core Endpoints

#### **Camera Capture**
```http
POST /api/capture
Content-Type: application/json

{
  "session_id": "session_12345",
  "camera_type": "pc|phone|ar",
  "image_data": "base64_encoded_image"
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "session_12345",
  "image_path": "/uploads/session_12345/captured_image.jpg",
  "audit_id": "audit_67890"
}
```

#### **Image Preprocessing**
```http
POST /api/preprocess
Content-Type: application/json

{
  "session_id": "session_12345",
  "image_data": "base64_encoded_image",
  "preprocessing_options": {
    "enhance_contrast": true,
    "correct_perspective": true,
    "reduce_noise": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "session_12345",
  "processed_image_path": "/processed/session_12345/enhanced_image.jpg",
  "preprocessing_info": {
    "operations_applied": ["contrast_enhancement", "perspective_correction"],
    "quality_metrics": {
      "sharpness": 145.2,
      "contrast": 67.8,
      "brightness": 128.5
    }
  }
}
```

#### **OMR Evaluation**
```http
POST /api/evaluate
Content-Type: application/json

{
  "session_id": "session_12345",
  "answer_key": {
    "1": "A", "2": "B", "3": "C", "4": "D", "5": "A"
  }
}
```

**Response:**
```json
{
  "success": true,
  "session_id": "session_12345",
  "evaluation_results": {
    "total_bubbles_detected": 45,
    "marked_bubbles": 25,
    "student_answers": {
      "1": {"answer": "A", "confidence": 0.95, "multiple_marks": false},
      "2": {"answer": "B", "confidence": 0.87, "multiple_marks": false}
    },
    "scoring_results": {
      "correct_answers": 18,
      "total_questions": 25,
      "score_percentage": 72.0,
      "grade": "B"
    }
  },
  "annotated_image_path": "/reports/session_12345/annotated_result.jpg"
}
```

#### **Batch Processing**
```http
POST /api/batch_process
Content-Type: multipart/form-data

files: [file1.jpg, file2.jpg, file3.jpg]
answer_key: {"1": "A", "2": "B", ...}
```

#### **System Health Check**
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "camera_service": "operational",
    "yolo_model": "loaded",
    "preprocessing": "ready",
    "audit_system": "active"
  },
  "system_info": {
    "uptime": "2 hours 15 minutes",
    "memory_usage": "45%",
    "storage_available": "12.5 GB"
  }
}
```

### Web Interface Routes

- **`/`** - Homepage with feature overview
- **`/camera`** - Camera capture interface  
- **`/dashboard`** - Analytics and monitoring dashboard
- **`/upload`** - File upload interface
- **`/results/<session_id>`** - View evaluation results
- **`/batch`** - Batch processing interface
- **`/audit`** - Audit trail viewer

## 🔧 Configuration Guide

### Environment Variables

```bash
# Flask Configuration
FLASK_ENV=development|production
FLASK_DEBUG=True|False
SECRET_KEY=your_secret_key_here

# SSL Configuration
SSL_CERT_PATH=ssl_certs/cert.pem
SSL_KEY_PATH=ssl_certs/key.pem

# Model Configuration
YOLO_MODEL_PATH=backend/yolov8n-cls.pt
CONFIDENCE_THRESHOLD=0.5

# Storage Configuration  
UPLOAD_FOLDER=uploads
PROCESSED_FOLDER=processed
REPORTS_FOLDER=reports
AUDIT_FOLDER=audit_logs

# Camera Configuration
DEFAULT_CAMERA_RESOLUTION=1920x1080
CAPTURE_FORMAT=JPEG
CAPTURE_QUALITY=90
```

### config.py Settings

```python
class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    
    # SSL settings for HTTPS
    SSL_CERT_PATH = 'ssl_certs/cert.pem'
    SSL_KEY_PATH = 'ssl_certs/key.pem'
    
    # Model settings
    YOLO_MODEL_PATH = '../backend/yolov8n-cls.pt'
    CONFIDENCE_THRESHOLD = 0.5
    
    # Storage settings
    UPLOAD_FOLDER = 'uploads'
    PROCESSED_FOLDER = 'processed'
    REPORTS_FOLDER = 'reports'
    AUDIT_FOLDER = 'audit_logs'
    
    # Camera settings
    CAMERA_RESOLUTION = (1920, 1080)
    CAPTURE_FORMAT = 'JPEG'
    CAPTURE_QUALITY = 90
```

## 🎛️ Service Components

### Camera Service (`services/camera_service.py`)

**Capabilities:**
- Multi-device camera detection and management
- PC webcam, phone camera, AR device support
- Automatic image enhancement and quality optimization
- Camera switching and property management

**Key Methods:**
```python
camera_service = CameraService()

# Get available cameras
cameras = camera_service.get_available_cameras()

# Capture image with enhancement
image_path = camera_service.capture_image(
    camera_index=0,
    camera_type='pc',
    output_path='captures/image.jpg',
    enhance=True
)
```

### Preprocessing Service (`services/preprocessing_service.py`)

**Capabilities:**
- Advanced noise reduction and filtering
- Perspective correction and geometric transformations
- Lighting normalization and contrast enhancement
- OMR-specific optimizations for bubble detection
- Quality assessment and metrics

**Key Methods:**
```python
preprocessing_service = PreprocessingService()

# Full preprocessing pipeline
result = preprocessing_service.process_image(
    image_path='input.jpg',
    output_path='processed.jpg',
    operations=['denoise', 'enhance_contrast', 'correct_perspective']
)

# Quality assessment
quality = preprocessing_service.assess_image_quality(image_path)
```

### YOLO Service (`services/yolo_service.py`)

**Capabilities:**
- YOLOv8 model loading and inference
- Bubble detection with confidence scoring
- Computer vision fallback for reliability
- Answer extraction and organization
- Automated scoring against answer keys
- Visual annotation and result reporting

**Key Methods:**
```python
yolo_service = YOLOService(model_path='yolov8n-cls.pt')

# Evaluate OMR sheet
results = yolo_service.evaluate_omr_sheet(
    image_path='preprocessed.jpg',
    answer_key={'1': 'A', '2': 'B', '3': 'C'},
    confidence_threshold=0.5
)

# Generate annotated image
annotated_path = yolo_service.generate_annotated_image(
    results, output_path='annotated.jpg'
)
```

### Audit Service (`services/audit_service.py`)

**Capabilities:**
- Complete session tracking and audit trails
- Compliance logging and data retention
- System statistics and performance metrics
- Audit report generation and export
- Data cleanup and retention management

**Key Methods:**
```python
audit_service = AuditService(audit_dir='audit_logs')

# Create audit entry
audit_entry = audit_service.create_capture_entry(
    session_id='session_123',
    camera_type='pc',
    image_path='captured.jpg',
    image_shape=(1080, 1920, 3)
)

# Get system statistics
stats = audit_service.get_system_statistics()

# Export audit report
report_path = audit_service.export_audit_report(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

## 🖥️ User Interface Guide

### Homepage (`/`)
- **System status overview** with real-time health indicators
- **Feature cards** showcasing all capabilities
- **Quick action buttons** for common tasks
- **System statistics** display (sessions, success rate, scores)

### Camera Interface (`/camera`)
- **Multi-device camera selection** (PC, Phone, AR)
- **Live video preview** with capture overlay
- **Camera controls** (start/stop, switch, capture)
- **Real-time image information** and quality metrics
- **Direct processing pipeline** integration

### Dashboard (`/dashboard`)
- **Real-time system analytics** and performance metrics
- **Interactive charts** showing activity and score distributions
- **Recent activity feed** with session summaries
- **Advanced filtering** and data visualization
- **Export and reporting** capabilities

## 🔒 Security Features

### HTTPS/SSL Implementation
- **Self-signed certificates** for development
- **Secure camera access** from web browsers
- **Certificate generation** automation
- **SSL context configuration** in Flask

### Data Protection
- **Session-based isolation** for multi-user support
- **Audit trail encryption** (configurable)
- **Secure file handling** with validation
- **Input sanitization** and validation

### Access Control
- **Session management** and tracking
- **API rate limiting** (configurable)
- **Error handling** without information disclosure
- **Secure file upload** with type validation

## 📊 Performance Optimization

### Image Processing
- **Efficient OpenCV operations** with memory optimization
- **Multi-threading support** for batch processing
- **Quality-based processing levels** (fast/balanced/high-quality)
- **Memory management** for large images

### Model Inference
- **GPU acceleration** support (when available)
- **Model caching** and persistent loading
- **Batch inference** optimization
- **Fallback mechanisms** for reliability

### Web Interface
- **Asynchronous processing** with progress tracking
- **Client-side image compression** before upload
- **Real-time updates** via WebSocket (future enhancement)
- **Caching strategies** for static assets

## 🧪 Testing and Quality Assurance

### Unit Testing
```bash
# Run individual service tests
python -m pytest tests/test_camera_service.py
python -m pytest tests/test_preprocessing_service.py
python -m pytest tests/test_yolo_service.py
python -m pytest tests/test_audit_service.py
```

### Integration Testing
```bash
# Full workflow testing
python -m pytest tests/test_integration.py

# API endpoint testing
python -m pytest tests/test_api_endpoints.py
```

### Manual Testing Checklist

1. **Camera Functionality**
   - [ ] PC webcam detection and capture
   - [ ] Phone camera access via browser
   - [ ] Camera switching (if multiple available)
   - [ ] Image quality and resolution

2. **Image Processing**
   - [ ] Preprocessing pipeline effectiveness
   - [ ] Quality assessment accuracy
   - [ ] Processing time performance
   - [ ] Output image quality

3. **OMR Evaluation**
   - [ ] Bubble detection accuracy
   - [ ] Answer extraction correctness
   - [ ] Scoring algorithm validation
   - [ ] Confidence metrics reliability

4. **User Interface**
   - [ ] Responsive design across devices
   - [ ] Real-time updates and feedback
   - [ ] Error handling and messaging
   - [ ] Navigation and usability

## 🚀 Deployment Guide

### Development Deployment
1. **Local Setup**
   ```bash
   python app.py
   # Access: https://localhost:5000
   ```

2. **Network Access**
   ```bash
   # Allow network access for mobile testing
   python app.py --host=0.0.0.0 --port=5000
   # Access: https://[your-ip]:5000
   ```

### Production Deployment

#### Option 1: Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

#### Option 2: Gunicorn + Nginx
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app

# Nginx configuration for SSL termination and reverse proxy
```

#### Option 3: Cloud Platform Deployment
- **AWS Elastic Beanstalk**: Upload ZIP with requirements.txt
- **Google Cloud Run**: Use containerized deployment
- **Azure App Service**: Python web app deployment
- **Heroku**: Git-based deployment with Procfile

### Environment Configuration

**Production Settings:**
```python
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    
    # Use environment variables for sensitive data
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Production SSL certificates
    SSL_CERT_PATH = os.environ.get('SSL_CERT_PATH')
    SSL_KEY_PATH = os.environ.get('SSL_KEY_PATH')
    
    # Enhanced security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # Production database (if using)
    DATABASE_URL = os.environ.get('DATABASE_URL')
```

## 🔧 Troubleshooting Guide

### Common Issues

#### **Camera Access Problems**
```
Error: Camera not accessible or permission denied
```
**Solutions:**
1. Ensure HTTPS is enabled (required for camera access)
2. Check browser permissions for camera access
3. Verify SSL certificate is trusted
4. Try different browsers (Chrome, Firefox, Edge)

#### **YOLO Model Issues**
```
Error: Model file not found or corrupted
```
**Solutions:**
1. Re-download YOLO model: `wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov8n-cls.pt`
2. Verify model path in config.py
3. Check PyTorch installation: `pip install torch torchvision`
4. Ensure sufficient disk space for model files

#### **SSL Certificate Problems**
```
Error: SSL certificate verification failed
```
**Solutions:**
1. Generate new self-signed certificates
2. Trust the certificate in browser settings
3. Use proper CA-signed certificates for production
4. Check certificate file permissions

#### **Image Processing Failures**
```
Error: OpenCV operation failed or unsupported format
```
**Solutions:**
1. Install OpenCV with full support: `pip install opencv-python opencv-contrib-python`
2. Check image format compatibility (JPEG, PNG supported)
3. Verify sufficient memory for large images
4. Update NumPy and scikit-image versions

### Performance Optimization

#### **Slow Processing Times**
1. **Enable GPU acceleration** (if available)
2. **Reduce image resolution** for faster processing
3. **Use batch processing** for multiple images
4. **Optimize preprocessing pipeline** (disable unnecessary operations)

#### **High Memory Usage**
1. **Process images in chunks** for batch operations
2. **Clear memory after processing** with explicit garbage collection
3. **Use efficient data types** (uint8 for images)
4. **Limit concurrent processing** threads

#### **Web Interface Responsiveness**
1. **Enable client-side image compression**
2. **Use asynchronous processing** with progress updates
3. **Implement caching** for frequently accessed data
4. **Optimize static asset delivery**

## 📈 Future Enhancements

### Planned Features
- **Real-time WebSocket updates** for processing status
- **Database integration** for persistent storage
- **User authentication** and role-based access
- **Advanced analytics** with machine learning insights
- **Mobile app development** for native camera access
- **Cloud storage integration** (AWS S3, Google Cloud Storage)
- **RESTful API versioning** and documentation automation
- **Kubernetes deployment** configurations

### Research Areas
- **Advanced computer vision** techniques for bubble detection
- **Machine learning** for answer pattern recognition
- **Natural language processing** for written answer evaluation
- **Augmented reality** integration for real-time feedback
- **Edge computing** deployment for offline processing

## 🤝 Contributing

### Development Workflow
1. **Fork the repository** and create feature branch
2. **Follow coding standards** (PEP 8 for Python, ESLint for JavaScript)
3. **Add comprehensive tests** for new features
4. **Update documentation** for API changes
5. **Submit pull request** with detailed description

### Code Style Guidelines
- **Python**: Follow PEP 8, use type hints
- **JavaScript**: ES6+ features, modern async/await
- **HTML/CSS**: Semantic markup, responsive design
- **Comments**: Comprehensive docstrings and inline comments

## 📞 Support and Contact

### Documentation Resources
- **GitHub Repository**: [Link to repository]
- **API Documentation**: Available at `/api/docs` endpoint
- **Video Tutorials**: [Link to video guides]
- **Sample Data**: Available in `Theme 1 - Sample Data/` directory

### Community Support
- **Issues and Bug Reports**: Use GitHub Issues
- **Feature Requests**: Submit via GitHub Discussions
- **Technical Questions**: Stack Overflow with `omr-flask-app` tag

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

**© 2024 OMR Flask Application - Advanced AI-Powered Assessment System**

Built with ❤️ using Flask, OpenCV, YOLO, and Modern Web Technologies