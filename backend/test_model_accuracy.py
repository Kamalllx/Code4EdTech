#!/usr/bin/env python3
"""
Test Model Accuracy for Marked vs Unmarked Bubble Detection
"""

import os
import cv2
import numpy as np
from pathlib import Path
from omr_processor import OMRProcessor
import json

def test_bubble_classification():
    """Test the model's ability to classify marked vs unmarked bubbles"""
    print("🔍 Testing Model Accuracy for Marked vs Unmarked Detection")
    print("=" * 60)
    
    # Load the trained model
    model_path = 'simple_training/omr_bubble_classifier/weights/best.pt'
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return False
    
    try:
        processor = OMRProcessor(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test on marked bubbles
    print("\n📊 Testing Marked Bubbles:")
    marked_path = Path('simple_dataset/val/marked_bubble')
    marked_files = list(marked_path.glob('*.jpg'))[:5]  # Test first 5 files
    
    marked_correct = 0
    marked_total = 0
    
    for img_file in marked_files:
        try:
            image = cv2.imread(str(img_file))
            if image is not None:
                # The model should classify this as "marked"
                result = processor.classify_bubble(image)
                predicted_class = result.get('class', 'unknown')
                confidence = result.get('confidence', 0)
                
                print(f"  {img_file.name}: {predicted_class} (confidence: {confidence:.2f})")
                
                if predicted_class == 'marked_bubble':
                    marked_correct += 1
                marked_total += 1
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
    
    # Test on unmarked bubbles
    print("\n📊 Testing Unmarked Bubbles:")
    unmarked_path = Path('simple_dataset/val/unmarked_bubble')
    unmarked_files = list(unmarked_path.glob('*.jpg'))[:5]  # Test first 5 files
    
    unmarked_correct = 0
    unmarked_total = 0
    
    for img_file in unmarked_files:
        try:
            image = cv2.imread(str(img_file))
            if image is not None:
                # The model should classify this as "unmarked"
                result = processor.classify_bubble(image)
                predicted_class = result.get('class', 'unknown')
                confidence = result.get('confidence', 0)
                
                print(f"  {img_file.name}: {predicted_class} (confidence: {confidence:.2f})")
                
                if predicted_class == 'unmarked_bubble':
                    unmarked_correct += 1
                unmarked_total += 1
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
    
    # Calculate accuracy
    marked_accuracy = (marked_correct / marked_total * 100) if marked_total > 0 else 0
    unmarked_accuracy = (unmarked_correct / unmarked_total * 100) if unmarked_total > 0 else 0
    overall_accuracy = ((marked_correct + unmarked_correct) / (marked_total + unmarked_total) * 100) if (marked_total + unmarked_total) > 0 else 0
    
    print("\n" + "=" * 60)
    print("📈 ACCURACY RESULTS")
    print("=" * 60)
    print(f"Marked Bubbles: {marked_correct}/{marked_total} ({marked_accuracy:.1f}%)")
    print(f"Unmarked Bubbles: {unmarked_correct}/{unmarked_total} ({unmarked_accuracy:.1f}%)")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    
    if overall_accuracy >= 80:
        print("✅ Model is performing well!")
    elif overall_accuracy >= 60:
        print("⚠️  Model needs improvement")
    else:
        print("❌ Model needs retraining")
    
    return overall_accuracy >= 60

def test_full_omr_sheet():
    """Test the model on a full OMR sheet"""
    print("\n🔍 Testing Full OMR Sheet Processing")
    print("=" * 60)
    
    # Test on a sample OMR sheet
    test_image_path = 'dataset/Set A/Img1.jpeg'
    if not os.path.exists(test_image_path):
        print("❌ Test OMR sheet not found!")
        return False
    
    try:
        processor = OMRProcessor('simple_training/omr_bubble_classifier/weights/best.pt')
        image = cv2.imread(test_image_path)
        
        print(f"Processing: {test_image_path}")
        result = processor.process_omr_sheet(image, 'A')
        
        if result.get('success', False):
            bubbles = result.get('bubbles', [])
            print(f"✅ Successfully processed OMR sheet")
            print(f"   - Total bubbles detected: {len(bubbles)}")
            
            # Count marked vs unmarked
            marked_count = sum(1 for bubble in bubbles if bubble.get('is_filled', False))
            unmarked_count = len(bubbles) - marked_count
            
            print(f"   - Marked bubbles: {marked_count}")
            print(f"   - Unmarked bubbles: {unmarked_count}")
            
            return True
        else:
            print("❌ Failed to process OMR sheet")
            return False
            
    except Exception as e:
        print(f"❌ Error processing OMR sheet: {e}")
        return False

def main():
    """Run all accuracy tests"""
    print("🧪 OMR Model Accuracy Test")
    print("=" * 60)
    
    # Test 1: Bubble classification accuracy
    classification_success = test_bubble_classification()
    
    # Test 2: Full OMR sheet processing
    sheet_success = test_full_omr_sheet()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    if classification_success and sheet_success:
        print("🎉 Model is working correctly!")
        print("✅ Can detect marked vs unmarked bubbles")
        print("✅ Can process full OMR sheets")
    elif classification_success:
        print("⚠️  Model can classify bubbles but has issues with full sheets")
    elif sheet_success:
        print("⚠️  Model can process sheets but has classification issues")
    else:
        print("❌ Model needs improvement in both areas")
    
    return classification_success and sheet_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
