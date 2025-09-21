#!/usr/bin/env python3
"""
Test Enhanced OMR Pipeline
Complete test of the enhanced OMR processing system
"""

import os
import sys
import json
import time
from pathlib import Path
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import OMRDatabase
from enhanced_omr_processor import EnhancedOMRProcessor

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_enhanced_pipeline.log'),
            logging.StreamHandler()
        ]
    )

def test_database_connection():
    """Test database connection and initialization"""
    print("\n" + "="*50)
    print("🗄️  Testing Database Connection")
    print("="*50)
    
    try:
        db = OMRDatabase("test_omr.db")
        print("✅ Database connection successful")
        
        # Test adding a student
        student_id = db.add_student("TEST001", "Test Student", "test@example.com")
        print(f"✅ Student added with ID: {student_id}")
        
        # Test adding an exam
        subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "English"]
        answer_key = {
            "Q1": "A", "Q2": "B", "Q3": "C", "Q4": "D", "Q5": "A",
            "Q6": "B", "Q7": "C", "Q8": "D", "Q9": "A", "Q10": "B",
            "Q11": "A", "Q12": "B", "Q13": "C", "Q14": "D", "Q15": "A",
            "Q16": "B", "Q17": "C", "Q18": "D", "Q19": "A", "Q20": "B"
        }
        
        exam_id = db.add_exam("Test Exam", "2024-01-15", 20, subjects, answer_key)
        print(f"✅ Exam added with ID: {exam_id}")
        
        return db, student_id, exam_id, answer_key
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return None, None, None, None

def test_enhanced_processor():
    """Test enhanced OMR processor initialization"""
    print("\n" + "="*50)
    print("🤖 Testing Enhanced OMR Processor")
    print("="*50)
    
    model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using train.py")
        return None
    
    try:
        processor = EnhancedOMRProcessor(model_path)
        print("✅ Enhanced OMR processor initialized successfully")
        return processor
    except Exception as e:
        print(f"❌ Enhanced processor initialization failed: {e}")
        return None

def test_image_processing(processor, image_path):
    """Test image processing capabilities"""
    print("\n" + "="*50)
    print("🖼️  Testing Image Processing")
    print("="*50)
    
    if not Path(image_path).exists():
        print(f"❌ Test image not found: {image_path}")
        return False
    
    try:
        # Test preprocessing
        print("📸 Testing image preprocessing...")
        processed_image = processor.preprocess_image(image_path)
        print(f"✅ Image preprocessed successfully (shape: {processed_image.shape})")
        
        # Test question number detection
        print("🔢 Testing question number detection...")
        question_numbers = processor.detect_question_numbers(processed_image)
        print(f"✅ Detected {len(question_numbers)} question numbers")
        
        if question_numbers:
            print("📋 Question numbers found:")
            for q in question_numbers[:5]:  # Show first 5
                print(f"   - Question {q['number']} at ({q['center'][0]}, {q['center'][1]}) - Confidence: {q['confidence']}")
        
        # Test bubble detection
        print("⭕ Testing bubble detection...")
        bubbles = processor.detect_bubbles_enhanced(processed_image)
        print(f"✅ Detected {len(bubbles)} bubbles")
        
        # Test bubble-to-question mapping
        print("🔗 Testing bubble-to-question mapping...")
        question_mapping = processor.map_bubbles_to_questions(question_numbers, bubbles)
        print(f"✅ Mapped {len(question_mapping)} questions to bubbles")
        
        if question_mapping:
            print("📋 Question mapping sample:")
            for q_key, q_data in list(question_mapping.items())[:3]:  # Show first 3
                print(f"   - {q_key}: {len(q_data['bubbles'])} bubbles mapped")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def test_complete_pipeline(processor, db, student_id, exam_id, answer_key, image_path):
    """Test the complete processing pipeline"""
    print("\n" + "="*50)
    print("🔄 Testing Complete Pipeline")
    print("="*50)
    
    try:
        print("🚀 Starting complete OMR processing pipeline...")
        start_time = time.time()
        
        # Process and save
        result = processor.process_and_save(
            image_path,
            student_id,
            exam_id,
            answer_key,
            "Set A"
        )
        
        processing_time = time.time() - start_time
        
        if result.get('success', True):
            print("✅ Complete pipeline executed successfully!")
            print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
            
            # Display results
            print("\n📊 Processing Results:")
            print(f"   - Total Questions: {result.get('total_questions', 0)}")
            print(f"   - Correct Answers: {result.get('total_score', 0)}")
            print(f"   - Percentage: {result.get('percentage', 0):.1f}%")
            print(f"   - Model Confidence: {result.get('model_confidence', 0):.3f}")
            print(f"   - Questions Mapped: {result.get('questions_mapped', 0)}")
            print(f"   - Bubbles Detected: {result.get('bubbles_detected', 0)}")
            
            # Show subject scores
            subject_scores = result.get('subject_scores', {})
            if subject_scores:
                print("\n📚 Subject Scores:")
                for subject, score in subject_scores.items():
                    print(f"   - {subject}: {score}")
            
            # Show sample answers
            student_answers = result.get('student_answers', {})
            if student_answers:
                print("\n📝 Sample Student Answers:")
                for q_key, answer in list(student_answers.items())[:5]:
                    print(f"   - {q_key}: {answer}")
            
            return True
        else:
            print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints (if backend is running)"""
    print("\n" + "="*50)
    print("🌐 Testing API Endpoints")
    print("="*50)
    
    try:
        import requests
        
        # Test health check
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend API is running")
            health_data = response.json()
            print(f"   - Status: {health_data.get('status', 'unknown')}")
            print(f"   - Model loaded: {health_data.get('model_loaded', False)}")
            return True
        else:
            print(f"❌ Backend API not responding (Status: {response.status_code})")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Backend API not running. Start it with: python web_backend.py")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def cleanup_test_data():
    """Clean up test data"""
    print("\n" + "="*50)
    print("🧹 Cleaning Up Test Data")
    print("="*50)
    
    try:
        # Remove test database
        if Path("test_omr.db").exists():
            os.remove("test_omr.db")
            print("✅ Test database removed")
        
        # Remove test log file
        if Path("test_enhanced_pipeline.log").exists():
            os.remove("test_enhanced_pipeline.log")
            print("✅ Test log file removed")
        
        print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def main():
    """Main test function"""
    print("🎯 Enhanced OMR Pipeline Test Suite")
    print("=" * 60)
    
    setup_logging()
    
    # Test 1: Database connection
    db, student_id, exam_id, answer_key = test_database_connection()
    if not db:
        print("\n❌ Database test failed. Exiting.")
        return
    
    # Test 2: Enhanced processor
    processor = test_enhanced_processor()
    if not processor:
        print("\n❌ Enhanced processor test failed. Exiting.")
        return
    
    # Test 3: Image processing (if test image exists)
    test_image_path = "uploads/test_omr_sheet.jpg"  # You can place a test image here
    if Path(test_image_path).exists():
        image_processing_success = test_image_processing(processor, test_image_path)
        if image_processing_success:
            # Test 4: Complete pipeline
            pipeline_success = test_complete_pipeline(processor, db, student_id, exam_id, answer_key, test_image_path)
        else:
            print("\n⚠️  Skipping complete pipeline test due to image processing failure")
    else:
        print(f"\n⚠️  Test image not found at {test_image_path}")
        print("   Place a test OMR sheet image there to test image processing")
    
    # Test 5: API endpoints
    api_success = test_api_endpoints()
    
    # Summary
    print("\n" + "="*50)
    print("📋 Test Summary")
    print("="*50)
    print("✅ Database connection: PASSED")
    print("✅ Enhanced processor: PASSED")
    if Path(test_image_path).exists():
        print(f"✅ Image processing: {'PASSED' if 'image_processing_success' in locals() and image_processing_success else 'FAILED'}")
        print(f"✅ Complete pipeline: {'PASSED' if 'pipeline_success' in locals() and pipeline_success else 'FAILED'}")
    else:
        print("⚠️  Image processing: SKIPPED (no test image)")
        print("⚠️  Complete pipeline: SKIPPED (no test image)")
    print(f"✅ API endpoints: {'PASSED' if api_success else 'FAILED'}")
    
    print("\n🎉 Enhanced OMR Pipeline Test Suite Completed!")
    print("\n💡 Next Steps:")
    print("   1. Place a test OMR sheet image in uploads/test_omr_sheet.jpg")
    print("   2. Start the backend API: python web_backend.py")
    print("   3. Test the frontend integration")
    
    # Cleanup
    cleanup_test_data()

if __name__ == "__main__":
    main()
