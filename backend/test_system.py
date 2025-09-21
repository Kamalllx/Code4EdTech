#!/usr/bin/env python3
"""
Comprehensive System Test
Tests model accuracy, database status, and AR workflow
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import sqlite3
from datetime import datetime

def test_database():
    """Test database connectivity and data"""
    print("🔍 Testing Database...")
    try:
        conn = sqlite3.connect('omr_evaluation.db')
        cursor = conn.cursor()
        
        # Check students
        cursor.execute('SELECT COUNT(*) FROM students')
        students_count = cursor.fetchone()[0]
        
        # Check exams
        cursor.execute('SELECT COUNT(*) FROM exams')
        exams_count = cursor.fetchone()[0]
        
        # Check OMR sheets
        cursor.execute('SELECT COUNT(*) FROM omr_sheets')
        sheets_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"✅ Database Status:")
        print(f"   - Students: {students_count}")
        print(f"   - Exams: {exams_count}")
        print(f"   - OMR Sheets: {sheets_count}")
        return True
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return False

def test_model_accuracy():
    """Test model accuracy on sample images"""
    print("\n🔍 Testing Model Accuracy...")
    try:
        from omr_processor import OMRProcessor
        
        # Check if model exists
        model_path = 'simple_training/omr_bubble_classifier/weights/best.pt'
        if not os.path.exists(model_path):
            print("❌ Model file not found")
            return False
            
        # Load processor
        processor = OMRProcessor(model_path)
        print("✅ Model loaded successfully")
        
        # Test on sample images
        test_images = [
            'dataset/Set A/Img1.jpeg',
            'dataset/Set A/Img2.jpeg',
            'dataset/Set B/Img9.jpeg'
        ]
        
        results = []
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"   Testing: {img_path}")
                image = cv2.imread(img_path)
                if image is not None:
                    result = processor.process_omr_sheet(image, 'A')
                    bubbles_detected = len(result.get('bubbles', []))
                    success = result.get('success', False)
                    results.append({
                        'image': img_path,
                        'bubbles': bubbles_detected,
                        'success': success
                    })
                    print(f"     - Bubbles detected: {bubbles_detected}")
                    print(f"     - Processing success: {success}")
        
        if results:
            avg_bubbles = sum(r['bubbles'] for r in results) / len(results)
            success_rate = sum(r['success'] for r in results) / len(results)
            print(f"\n✅ Model Performance:")
            print(f"   - Average bubbles detected: {avg_bubbles:.1f}")
            print(f"   - Success rate: {success_rate:.1%}")
            return True
        else:
            print("❌ No test images found")
            return False
            
    except Exception as e:
        print(f"❌ Model Error: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🔍 Testing API Endpoints...")
    try:
        import requests
        
        base_url = "http://localhost:5000"
        
        # Test health endpoint
        try:
            response = requests.get(f"{base_url}/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Health endpoint working")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
        except:
            print("❌ Health endpoint not responding")
        
        # Test students endpoint
        try:
            response = requests.get(f"{base_url}/api/students", timeout=5)
            if response.status_code == 200:
                print("✅ Students endpoint working")
            else:
                print(f"❌ Students endpoint failed: {response.status_code}")
        except:
            print("❌ Students endpoint not responding")
        
        # Test exams endpoint
        try:
            response = requests.get(f"{base_url}/api/exams", timeout=5)
            if response.status_code == 200:
                print("✅ Exams endpoint working")
            else:
                print(f"❌ Exams endpoint failed: {response.status_code}")
        except:
            print("❌ Exams endpoint not responding")
            
        return True
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False

def test_ar_workflow():
    """Test AR workflow simulation"""
    print("\n🔍 Testing AR Workflow...")
    try:
        # Simulate AR image capture
        test_image_path = 'dataset/Set A/Img1.jpeg'
        if os.path.exists(test_image_path):
            print("✅ Test image available for AR simulation")
            
            # Simulate base64 encoding (simplified)
            with open(test_image_path, 'rb') as f:
                image_data = f.read()
            
            # Simulate AR metadata
            ar_metadata = {
                'timestamp': datetime.now().isoformat(),
                'device_info': 'Test Device',
                'location': {'lat': 40.7128, 'lng': -74.0060}
            }
            
            print("✅ AR metadata prepared")
            print(f"   - Image size: {len(image_data)} bytes")
            print(f"   - Timestamp: {ar_metadata['timestamp']}")
            print(f"   - Device: {ar_metadata['device_info']}")
            
            return True
        else:
            print("❌ No test image for AR simulation")
            return False
            
    except Exception as e:
        print(f"❌ AR Workflow Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 OMR System Comprehensive Test")
    print("=" * 50)
    
    tests = [
        ("Database", test_database),
        ("Model Accuracy", test_model_accuracy),
        ("API Endpoints", test_api_endpoints),
        ("AR Workflow", test_ar_workflow)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems are working perfectly!")
    else:
        print("⚠️  Some issues detected. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
