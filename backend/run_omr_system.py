#!/usr/bin/env python3
"""
OMR System Runner
Main script to run the complete OMR evaluation system
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'ultralytics',
        'opencv-python',
        'numpy',
        'pandas',
        'streamlit',
        'flask',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_model():
    """Check if trained model exists"""
    model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Trained model not found at: {model_path}")
        print("Please train the model first using: python train.py")
        return False
    
    print("✅ Trained model found")
    return True

def run_streamlit():
    """Run Streamlit web interface"""
    print("🚀 Starting Streamlit web interface...")
    print("📱 Open your browser and go to: http://localhost:8501")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_frontend.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit interface stopped")

def run_flask_api():
    """Run Flask API backend"""
    print("🚀 Starting Flask API backend...")
    print("🔗 API available at: http://localhost:5000")
    
    try:
        subprocess.run([sys.executable, "web_backend.py"])
    except KeyboardInterrupt:
        print("\n👋 Flask API stopped")

def run_training():
    """Run model training"""
    print("🚀 Starting model training...")
    
    # Check if dataset exists
    if not Path("dataset").exists():
        print("❌ Dataset directory not found")
        print("Please place your OMR images in the 'dataset' directory")
        return False
    
    try:
        subprocess.run([
            sys.executable, "train.py",
            "--data_dir", "dataset",
            "--output_dir", "omr_training_results",
            "--model_type", "classification",
            "--epochs", "50",
            "--batch_size", "16",
            "--augment"
        ])
        return True
    except KeyboardInterrupt:
        print("\n👋 Training stopped")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='OMR Evaluation System Runner')
    parser.add_argument('--mode', choices=['web', 'api', 'train', 'check'], 
                       default='web', help='Mode to run the system')
    parser.add_argument('--port', type=int, default=8501, 
                       help='Port for web interface (default: 8501)')
    parser.add_argument('--api-port', type=int, default=5000, 
                       help='Port for API backend (default: 5000)')
    
    args = parser.parse_args()
    
    print("🎯 OMR Evaluation System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    if args.mode == 'check':
        print("✅ System check completed")
        if check_model():
            print("🎉 System is ready to use!")
        else:
            print("⚠️  Please train the model first")
        return 0
    
    if args.mode == 'train':
        print("📚 Training mode selected")
        return 0 if run_training() else 1
    
    if args.mode == 'api':
        print("🔧 API mode selected")
        if not check_model():
            print("⚠️  Model not found. Please train first or use '--mode train'")
            return 1
        run_flask_api()
        return 0
    
    if args.mode == 'web':
        print("🌐 Web interface mode selected")
        if not check_model():
            print("⚠️  Model not found. Please train first or use '--mode train'")
            return 1
        
        print("\nChoose an option:")
        print("1. Run Streamlit web interface (recommended)")
        print("2. Run Flask API backend")
        print("3. Run both (API + Web)")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            run_streamlit()
        elif choice == '2':
            run_flask_api()
        elif choice == '3':
            print("🚀 Starting both API and Web interface...")
            print("📱 Web interface: http://localhost:8501")
            print("🔗 API backend: http://localhost:5000")
            print("Press Ctrl+C to stop both services")
            
            # This is a simplified version - in production you'd want proper process management
            try:
                # Start API in background
                api_process = subprocess.Popen([sys.executable, "web_backend.py"])
                
                # Start Streamlit
                subprocess.run([
                    sys.executable, "-m", "streamlit", "run", "web_frontend.py",
                    "--server.port", str(args.port)
                ])
            except KeyboardInterrupt:
                print("\n👋 Stopping services...")
                api_process.terminate()
        else:
            print("❌ Invalid choice")
            return 1
        
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)