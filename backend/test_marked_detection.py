#!/usr/bin/env python3
"""
Test if the model is correctly detecting marked answers
"""

import os
import cv2
import numpy as np
from pathlib import Path
from demo_bubble_classifier import OMRBubbleClassifier

def test_marked_vs_unmarked():
    """Test the model's ability to distinguish marked from unmarked bubbles"""
    print("🔍 Testing Model: Marked vs Unmarked Bubble Detection")
    print("=" * 60)
    
    # Load the trained model
    model_path = 'simple_training/omr_bubble_classifier/weights/best.pt'
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return False
    
    try:
        classifier = OMRBubbleClassifier(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test on marked bubbles
    print("\n📊 Testing Marked Bubbles:")
    marked_path = Path('simple_dataset/val/marked_bubble')
    marked_files = list(marked_path.glob('*.jpg'))[:3]  # Test first 3 files
    
    marked_correct = 0
    marked_total = 0
    
    for img_file in marked_files:
        try:
            result = classifier.classify_bubble(str(img_file), verbose=False)
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            print(f"  {img_file.name}: {predicted_class} (confidence: {confidence:.2f})")
            
            if predicted_class == 'marked_bubble':
                marked_correct += 1
            marked_total += 1
        except Exception as e:
            print(f"  Error processing {img_file.name}: {e}")
    
    # Test on unmarked bubbles
    print("\n📊 Testing Unmarked Bubbles:")
    unmarked_path = Path('simple_dataset/val/unmarked_bubble')
    unmarked_files = list(unmarked_path.glob('*.jpg'))[:3]  # Test first 3 files
    
    unmarked_correct = 0
    unmarked_total = 0
    
    for img_file in unmarked_files:
        try:
            result = classifier.classify_bubble(str(img_file), verbose=False)
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
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
        print("✅ Model is detecting marked answers correctly!")
    elif overall_accuracy >= 60:
        print("⚠️  Model is working but needs improvement")
    else:
        print("❌ Model is not detecting marked answers properly")
    
    return overall_accuracy >= 60

def test_confidence_scores():
    """Test the confidence scores for marked vs unmarked bubbles"""
    print("\n🔍 Testing Confidence Scores")
    print("=" * 60)
    
    try:
        classifier = OMRBubbleClassifier('simple_training/omr_bubble_classifier/weights/best.pt')
        
        # Test a few marked bubbles
        marked_path = Path('simple_dataset/val/marked_bubble')
        marked_files = list(marked_path.glob('*.jpg'))[:2]
        
        print("Marked Bubbles (should have high confidence for 'marked_bubble'):")
        for img_file in marked_files:
            result = classifier.classify_bubble(str(img_file), verbose=False)
            print(f"  {img_file.name}: {result['class']} (confidence: {result['confidence']:.3f})")
        
        # Test a few unmarked bubbles
        unmarked_path = Path('simple_dataset/val/unmarked_bubble')
        unmarked_files = list(unmarked_path.glob('*.jpg'))[:2]
        
        print("\nUnmarked Bubbles (should have high confidence for 'unmarked_bubble'):")
        for img_file in unmarked_files:
            result = classifier.classify_bubble(str(img_file), verbose=False)
            print(f"  {img_file.name}: {result['class']} (confidence: {result['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing confidence scores: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 OMR Model Marked Answer Detection Test")
    print("=" * 60)
    
    # Test 1: Basic accuracy
    accuracy_success = test_marked_vs_unmarked()
    
    # Test 2: Confidence scores
    confidence_success = test_confidence_scores()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    
    if accuracy_success:
        print("🎉 Model is correctly detecting marked answers!")
        print("✅ Can distinguish between marked and unmarked bubbles")
        if confidence_success:
            print("✅ Confidence scores are working properly")
    else:
        print("❌ Model is NOT detecting marked answers correctly")
        print("⚠️  The model may need retraining or the test data may be incorrect")
    
    return accuracy_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
