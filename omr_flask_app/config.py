"""
Configuration settings for OMR Flask Application
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = 'omr-flask-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Application directories
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    RESULTS_FOLDER = BASE_DIR / 'results'
    AUDIT_DIR = BASE_DIR / 'audit_trail'
    TEMPLATES_DIR = BASE_DIR / 'templates'
    STATIC_DIR = BASE_DIR / 'static'
    
    # YOLO Model settings
    YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH') or str(BASE_DIR.parent / 'backend' / 'yolov8n-cls.pt')
    MODEL_CONFIDENCE_THRESHOLD = float(os.environ.get('MODEL_CONFIDENCE', '0.5'))
    
    # Camera settings
    CAMERA_WIDTH = int(os.environ.get('CAMERA_WIDTH', '1280'))
    CAMERA_HEIGHT = int(os.environ.get('CAMERA_HEIGHT', '720'))
    CAMERA_FPS = int(os.environ.get('CAMERA_FPS', '30'))
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # Security settings
    CORS_ENABLED = True
    HTTPS_REQUIRED = os.environ.get('HTTPS_REQUIRED', 'True').lower() == 'true'
    
    # Preprocessing settings
    PREPROCESSING_CONFIG = {
        'adaptive_threshold': True,
        'morphological_operations': True,
        'noise_reduction': True,
        'contrast_enhancement': True,
        'resize_target': (1024, 768),  # Target size for processing
        'blur_kernel_size': 3,
        'morphology_kernel_size': 3
    }
    
    # Audit settings
    AUDIT_RETENTION_DAYS = int(os.environ.get('AUDIT_RETENTION_DAYS', '30'))
    ENABLE_DETAILED_LOGGING = os.environ.get('DETAILED_LOGGING', 'True').lower() == 'true'
    
    # OMR Evaluation settings
    OMR_CONFIG = {
        'bubble_detection': {
            'min_radius': 10,
            'max_radius': 25,
            'dp': 1,
            'min_dist': 20,
            'param1': 50,
            'param2': 30
        },
        'marking_threshold': 0.3,  # Darkness ratio for marked bubbles
        'confidence_threshold': 0.5
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    HTTPS_REQUIRED = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    HTTPS_REQUIRED = True
    SECRET_KEY = os.environ.get('SECRET_KEY', 'omr-flask-secret-key-change-in-production')
    
    # Only raise error if SECRET_KEY is explicitly required to be from environment
    # For development/testing, we'll allow the default key

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    HTTPS_REQUIRED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}