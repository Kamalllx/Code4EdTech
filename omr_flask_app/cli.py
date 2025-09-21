#!/usr/bin/env python3
"""
CLI Interface for OMR Flask Application
Provides command-line interaction with the OMR evaluation system
"""

import argparse
import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import time
from datetime import datetime
import base64
from urllib.parse import urljoin
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available - Excel answer keys will not be supported")

# Add the Flask app directory to Python path
sys.path.append(str(Path(__file__).parent))

from services.camera_service import CameraService
from services.preprocessing_service import PreprocessingService
from services.yolo_service import YOLOService
from services.audit_service import AuditService
from config import Config


def parse_excel_answer_key(excel_path: str) -> Dict[str, Any]:
    """
    Parse Excel answer key in the format of Key (Set A and B).xlsx
    
    Returns a standardized answer key format
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required to parse Excel answer keys")
    
    try:
        df = pd.read_excel(excel_path)
        
        # Parse the answer key structure
        answer_key = {
            "exam_id": "excel_answer_key",
            "exam_title": f"Answer Key from {Path(excel_path).name}",
            "total_questions": 0,
            "correct_answers": {},
            "sections": {},
            "scoring": {
                "correct_marks": 1,
                "incorrect_marks": -0.25,
                "unanswered_marks": 0,
                "grading_scale": {
                    "A+": 90, "A": 80, "B+": 70, "B": 60, "C": 50, "D": 40, "F": 0
                }
            }
        }
        
        question_num = 1
        
        # Process each column (subject/section)
        for col_name in df.columns:
            col_name_clean = col_name.strip()
            section_answers = {}
            
            for idx, row in df.iterrows():
                cell_value = str(row[col_name]).strip()
                
                # Handle both formats: "1 - a" and "81. a"
                if ' - ' in cell_value or '. ' in cell_value:
                    # Normalize format by replacing '. ' with ' - '
                    normalized_value = cell_value.replace('. ', ' - ')
                    parts = normalized_value.split(' - ')
                    
                    if len(parts) >= 2:
                        q_num_str = parts[0].strip()
                        answer = parts[1].strip().upper()
                        
                        # Extract question number (handle cases like "81." or "1")
                        q_num_digits = ''.join(filter(str.isdigit, q_num_str))
                        if q_num_digits:
                            q_num = int(q_num_digits)
                            
                            section_answers[str(q_num)] = answer
                            answer_key["correct_answers"][str(question_num)] = answer
                            question_num += 1
            
            answer_key["sections"][col_name_clean] = section_answers
        
        answer_key["total_questions"] = question_num - 1
        
        print(f"📋 Parsed Excel answer key: {answer_key['total_questions']} questions from {len(answer_key['sections'])} sections")
        for section, answers in answer_key["sections"].items():
            print(f"   📖 {section}: {len(answers)} questions")
        
        return answer_key
        
    except Exception as e:
        raise ValueError(f"Failed to parse Excel answer key: {str(e)}")


class OMRCLIClient:
    """CLI client for OMR Flask application"""
    
    def __init__(self, base_url: str = "http://localhost:5000", offline_mode: bool = False):
        """
        Initialize CLI client
        
        Args:
            base_url: Base URL of Flask application
            offline_mode: Use local services instead of API calls
        """
        self.base_url = base_url
        self.offline_mode = offline_mode
        self.session_id = f"cli_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create output directories
        self.output_dir = Path("cli_output")
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "captures").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Auto-detect offline mode if Flask app is not available
        if not self.offline_mode:
            try:
                response = requests.get(f"{self.base_url}/api/health", timeout=3, verify=False)
                if response.status_code != 200:
                    print("⚠️  Flask app not responding, switching to offline mode")
                    self.offline_mode = True
            except Exception:
                print("⚠️  Flask app not available, switching to offline mode")
                self.offline_mode = True
        
        # Initialize offline services if needed
        if self.offline_mode:
            self._init_offline_services()
    
    def _init_offline_services(self):
        """Initialize offline services"""
        try:
            config = Config()
            self.camera_service = CameraService()
            self.preprocessing_service = PreprocessingService()
            
            # Check if YOLO model exists
            model_path = Path(config.YOLO_MODEL_PATH)
            if model_path.exists():
                self.yolo_service = YOLOService(str(model_path))
                print(f"✅ YOLO model loaded from: {model_path}")
            else:
                print(f"⚠️  YOLO model not found at: {model_path}")
                self.yolo_service = None
            
            self.audit_service = AuditService(str(self.output_dir / "audit"))
            print("✅ Offline services initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing offline services: {str(e)}")
            self.offline_mode = False
    
    def _make_api_call(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make API call to Flask application"""
        if self.offline_mode:
            return {"error": "Offline mode - API calls not available"}
        
        try:
            url = urljoin(self.base_url, endpoint)
            
            if method == "GET":
                response = requests.get(url, verify=False, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=data, verify=False, timeout=30)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {str(e)}"}
    
    def capture_image(self, camera_type: str = "pc", output_path: Optional[str] = None) -> Dict[str, Any]:
        """Capture image using camera"""
        print(f"📷 Capturing image using {camera_type} camera...")
        
        if not output_path:
            output_path = self.output_dir / "captures" / f"capture_{self.session_id}.jpg"
        
        if self.offline_mode and hasattr(self, 'camera_service'):
            try:
                # Use offline camera service
                camera_index = 0  # Default to first camera
                captured_path = self.camera_service.capture_image(
                    camera_index=camera_index,
                    camera_type=camera_type,
                    output_path=str(output_path),
                    enhance=True
                )
                
                if captured_path and os.path.exists(captured_path):
                    image = cv2.imread(captured_path)
                    
                    # Create audit entry
                    audit_entry = self.audit_service.create_capture_entry(
                        session_id=self.session_id,
                        camera_type=camera_type,
                        image_path=captured_path,
                        image_shape=image.shape
                    )
                    
                    print(f"✅ Image captured successfully: {captured_path}")
                    return {
                        "success": True,
                        "image_path": captured_path,
                        "audit_id": audit_entry["audit_id"]
                    }
                else:
                    return {"success": False, "error": "Failed to capture image"}
                    
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            # Use API call (placeholder - would need base64 image handling)
            return self._make_api_call("/api/capture", "POST", {
                "session_id": self.session_id,
                "camera_type": camera_type
            })
    
    def process_file(self, file_path: str, answer_key_file: Optional[str] = None) -> Dict[str, Any]:
        """Process an existing OMR image file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}
        
        print(f"📄 Processing file: {file_path}")
        
        try:
            # Load image
            image = cv2.imread(str(file_path))
            if image is None:
                return {"success": False, "error": "Failed to load image"}
            
            # Load answer key if provided
            answer_key = None
            if answer_key_file:
                answer_key_path = Path(answer_key_file)
                if answer_key_path.exists():
                    # Check if it's an Excel file
                    if answer_key_path.suffix.lower() in ['.xlsx', '.xls']:
                        try:
                            answer_key = parse_excel_answer_key(str(answer_key_path))
                            print(f"📋 Loaded Excel answer key: {answer_key_path}")
                        except Exception as e:
                            print(f"❌ Failed to parse Excel answer key: {str(e)}")
                            return {"success": False, "error": f"Excel parsing failed: {str(e)}"}
                    else:
                        # JSON format
                        try:
                            with open(answer_key_path, 'r') as f:
                                answer_key = json.load(f)
                            print(f"📋 Loaded JSON answer key: {answer_key_path}")
                        except Exception as e:
                            print(f"❌ Failed to parse JSON answer key: {str(e)}")
                            return {"success": False, "error": f"JSON parsing failed: {str(e)}"}
                else:
                    print(f"⚠️  Answer key file not found: {answer_key_path}")
            
            # Process the image
            return self._process_image(image, str(file_path), answer_key)
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _process_image(self, image: np.ndarray, source_path: str, answer_key: Optional[Dict] = None) -> Dict[str, Any]:
        """Process image through the OMR pipeline"""
        results = {"success": False, "stages": {}}
        
        try:
            # Stage 1: Preprocessing
            print("🔧 Step 1: Preprocessing image...")
            
            if self.offline_mode and hasattr(self, 'preprocessing_service'):
                try:
                    # Use the correct method from preprocessing service
                    processed_image, preprocessing_info = self.preprocessing_service.preprocess_omr_image(image)
                    
                    # Save preprocessed image
                    preprocessed_path = self.output_dir / "processed" / f"preprocessed_{self.session_id}.jpg"
                    cv2.imwrite(str(preprocessed_path), processed_image)
                    
                    results["stages"]["preprocessing"] = {
                        'success': True,
                        'preprocessing_info': preprocessing_info,
                        'preprocessed_path': str(preprocessed_path)
                    }
                    print("✅ Preprocessing completed")
                    
                except Exception as e:
                    print(f"❌ Preprocessing failed: {str(e)}")
                    processed_image = image
                    results["stages"]["preprocessing"] = {"success": False, "error": str(e)}
            else:
                processed_image = image
                results["stages"]["preprocessing"] = {"success": True, "note": "Skipped preprocessing"}
            
            # Stage 2: OMR Evaluation
            print("🎯 Step 2: OMR Evaluation...")
            
            if self.offline_mode and self.yolo_service:
                evaluation_result = self.yolo_service.evaluate_omr_sheet(processed_image, answer_key)
                results["stages"]["evaluation"] = evaluation_result
                
                if evaluation_result['success']:
                    print(f"✅ OMR Evaluation completed")
                    print(f"   📊 Bubbles detected: {evaluation_result.get('total_bubbles_detected', 0)}")
                    print(f"   ✏️  Marked bubbles: {evaluation_result.get('marked_bubbles', 0)}")
                    
                    # Generate annotated image
                    annotated_image = self.yolo_service.generate_annotated_image(processed_image, evaluation_result)
                    annotated_path = self.output_dir / "results" / f"annotated_{self.session_id}.jpg"
                    cv2.imwrite(str(annotated_path), annotated_image)
                    results["annotated_image_path"] = str(annotated_path)
                    
                    # Display scores if available
                    scoring_results = evaluation_result.get('scoring_results', {})
                    if scoring_results:
                        score = scoring_results.get('score_percentage', 0)
                        correct = scoring_results.get('correct_answers', 0)
                        total = scoring_results.get('total_questions', 0)
                        grade = scoring_results.get('grade', 'N/A')
                        
                        print(f"   🏆 Score: {correct}/{total} ({score:.1f}%) - Grade: {grade}")
                    
                else:
                    print(f"❌ OMR Evaluation failed: {evaluation_result.get('error', 'Unknown error')}")
            else:
                print("⚠️  YOLO service not available - skipping evaluation")
                results["stages"]["evaluation"] = {"success": False, "error": "YOLO service not available"}
            
            # Stage 3: Generate Report
            print("📄 Step 3: Generating report...")
            
            report_data = {
                "session_id": self.session_id,
                "source_file": source_path,
                "processing_timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            report_path = self.output_dir / "results" / f"report_{self.session_id}.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            results["report_path"] = str(report_path)
            results["success"] = True
            
            print(f"✅ Processing completed successfully")
            print(f"📄 Report saved: {report_path}")
            
            if "annotated_image_path" in results:
                print(f"🖼️  Annotated image: {results['annotated_image_path']}")
            
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            print(f"❌ Processing failed: {str(e)}")
        
        return results
    
    def batch_process(self, input_dir: str, answer_key_file: Optional[str] = None, 
                     file_pattern: str = "*.jpg") -> Dict[str, Any]:
        """Process multiple OMR images in batch"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            return {"success": False, "error": f"Input directory not found: {input_path}"}
        
        # Find image files
        image_files = list(input_path.glob(file_pattern))
        if not image_files:
            return {"success": False, "error": f"No images found with pattern: {file_pattern}"}
        
        print(f"📁 Batch processing {len(image_files)} images from: {input_path}")
        
        # Load answer key if provided
        answer_key = None
        if answer_key_file:
            try:
                answer_key_path = Path(answer_key_file)
                if answer_key_path.suffix.lower() in ['.xlsx', '.xls']:
                    answer_key = parse_excel_answer_key(str(answer_key_path))
                    print(f"📋 Using Excel answer key: {answer_key_file}")
                else:
                    with open(answer_key_file, 'r') as f:
                        answer_key = json.load(f)
                    print(f"📋 Using JSON answer key: {answer_key_file}")
            except Exception as e:
                print(f"⚠️  Failed to load answer key: {str(e)}")
        
        # Process each image
        batch_results = []
        successful_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"\n--- Processing {i}/{len(image_files)}: {image_file.name} ---")
            
            # Create unique session ID for each image
            original_session_id = self.session_id
            self.session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"
            
            result = self.process_file(str(image_file), answer_key_file)
            result["file_name"] = image_file.name
            result["file_index"] = i
            
            batch_results.append(result)
            
            if result.get("success", False):
                successful_count += 1
                print(f"✅ {image_file.name} processed successfully")
            else:
                print(f"❌ {image_file.name} failed: {result.get('error', 'Unknown error')}")
            
            # Restore original session ID
            self.session_id = original_session_id
        
        # Generate batch summary
        batch_summary = {
            "total_files": len(image_files),
            "successful": successful_count,
            "failed": len(image_files) - successful_count,
            "success_rate": (successful_count / len(image_files)) * 100,
            "processing_timestamp": datetime.now().isoformat(),
            "results": batch_results
        }
        
        # Save batch report
        batch_report_path = self.output_dir / "results" / f"batch_report_{self.session_id}.json"
        with open(batch_report_path, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total files: {batch_summary['total_files']}")
        print(f"   Successful: {batch_summary['successful']}")
        print(f"   Failed: {batch_summary['failed']}")
        print(f"   Success rate: {batch_summary['success_rate']:.1f}%")
        print(f"📄 Batch report: {batch_report_path}")
        
        return batch_summary
    
    def system_status(self) -> Dict[str, Any]:
        """Check system status"""
        print("🔍 Checking system status...")
        
        if self.offline_mode:
            status = {
                "mode": "offline",
                "camera_service": hasattr(self, 'camera_service'),
                "preprocessing_service": hasattr(self, 'preprocessing_service'),
                "yolo_service": self.yolo_service is not None,
                "audit_service": hasattr(self, 'audit_service')
            }
            
            if self.yolo_service:
                model_status = self.yolo_service.check_model_status()
                status["model_info"] = model_status
            
            # Check output directories
            status["output_directories"] = {
                "main": self.output_dir.exists(),
                "captures": (self.output_dir / "captures").exists(),
                "processed": (self.output_dir / "processed").exists(),
                "results": (self.output_dir / "results").exists()
            }
            
        else:
            # Check Flask application status
            status = self._make_api_call("/api/health")
        
        return status
    
    def list_cameras(self) -> List[Dict[str, Any]]:
        """List available cameras"""
        if self.offline_mode and hasattr(self, 'camera_service'):
            cameras = self.camera_service.get_available_cameras()
            print(f"📷 Found {len(cameras)} available cameras:")
            for i, camera in enumerate(cameras):
                print(f"   Camera {i}: {camera}")
            return cameras
        else:
            print("⚠️  Camera listing only available in offline mode")
            return []


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(
        description="OMR Flask Application CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s capture                              # Capture image from PC camera
  %(prog)s capture --camera phone               # Capture from phone camera
  %(prog)s process image.jpg                    # Process single image
  %(prog)s process image.jpg --answer-key key.json  # Process with answer key
  %(prog)s batch images/ --answer-key key.json  # Batch process directory
  %(prog)s status                               # Check system status
  %(prog)s cameras                              # List available cameras
        """
    )
    
    parser.add_argument('--url', default='http://localhost:5000',
                       help='Flask application URL (default: http://localhost:5000)')
    parser.add_argument('--offline', action='store_true',
                       help='Use offline mode (local services only)')
    parser.add_argument('--output-dir', default='cli_output',
                       help='Output directory for results (default: cli_output)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Capture command
    capture_parser = subparsers.add_parser('capture', help='Capture image from camera')
    capture_parser.add_argument('--camera', choices=['pc', 'phone', 'ar'], default='pc',
                               help='Camera type (default: pc)')
    capture_parser.add_argument('--output', help='Output image path')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process OMR image file')
    process_parser.add_argument('file', help='Image file to process')
    process_parser.add_argument('--answer-key', help='Answer key JSON file')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple images')
    batch_parser.add_argument('directory', help='Directory containing images')
    batch_parser.add_argument('--answer-key', help='Answer key JSON file')
    batch_parser.add_argument('--pattern', default='*.jpg',
                             help='File pattern to match (default: *.jpg)')
    
    # Status command
    subparsers.add_parser('status', help='Check system status')
    
    # Cameras command
    subparsers.add_parser('cameras', help='List available cameras')
    
    return parser


def main():
    """Main CLI entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Print banner
    print("=" * 60)
    print("🎯 OMR Flask Application - CLI Interface")
    print("   Advanced Optical Mark Recognition System")
    print("=" * 60)
    
    # Initialize CLI client
    try:
        client = OMRCLIClient(base_url=args.url, offline_mode=args.offline)
        
        if args.offline:
            print("🔧 Running in OFFLINE mode (using local services)")
        else:
            print(f"🌐 Connecting to Flask application: {args.url}")
            if not client.offline_mode:
                print("✅ Flask app connection successful")
            else:
                print("⚠️  Switched to offline mode due to connection issues")
        
    except Exception as e:
        print(f"❌ Failed to initialize CLI client: {str(e)}")
        return 1
    
    # Execute command
    try:
        if args.command == 'capture':
            result = client.capture_image(
                camera_type=args.camera,
                output_path=args.output
            )
            
            if result.get("success", False):
                print(f"🎉 Capture successful!")
                if "image_path" in result:
                    print(f"📸 Image saved: {result['image_path']}")
            else:
                print(f"❌ Capture failed: {result.get('error', 'Unknown error')}")
                return 1
        
        elif args.command == 'process':
            result = client.process_file(args.file, args.answer_key)
            
            if result.get("success", False):
                print(f"🎉 Processing successful!")
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                return 1
        
        elif args.command == 'batch':
            result = client.batch_process(args.directory, args.answer_key, args.pattern)
            
            if result.get("success_rate", 0) > 0:
                print(f"🎉 Batch processing completed!")
            else:
                print(f"❌ Batch processing had issues")
                return 1
        
        elif args.command == 'status':
            status = client.system_status()
            print("\n📊 System Status:")
            print(json.dumps(status, indent=2, default=str))
        
        elif args.command == 'cameras':
            cameras = client.list_cameras()
            if not cameras:
                print("⚠️  No cameras found or camera listing not available")
        
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return 1
    
    print("\n✅ CLI operation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())