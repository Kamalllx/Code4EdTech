#!/usr/bin/env python3
"""
OMR Bubble Classification - Demo Script
Demonstrates how to use the trained model for bubble classification.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json

class OMRBubbleClassifier:
    """Simple wrapper for the trained OMR bubble classification model"""
    
    def __init__(self, model_path):
        """Initialize the classifier with the trained model"""
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading OMR Bubble Classification Model...")
        self.model = YOLO(str(self.model_path))
        print(f"✅ Model loaded successfully from: {model_path}")
        
        # Get class names from model
        self.class_names = {0: 'marked_bubble', 1: 'unmarked_bubble'}
        
    def classify_bubble(self, image_input, verbose=True):
        """
        Classify a single bubble image
        
        Args:
            image_input: Can be:
                - String path to image file
                - NumPy array (H, W, 3)
                - OpenCV image
            verbose: Print prediction details
            
        Returns:
            dict: Classification results
        """
        
        # Run inference
        results = self.model(image_input, verbose=False)
        result = results[0]
        
        # Extract prediction details
        predicted_class_id = result.probs.top1
        predicted_class_name = self.class_names[predicted_class_id]
        confidence = result.probs.top1conf.item()
        all_probabilities = result.probs.data.tolist()
        
        # Create result dictionary
        classification_result = {
            'predicted_class': predicted_class_name,
            'class_id': predicted_class_id,
            'confidence': confidence,
            'confidence_percent': f"{confidence:.1%}",
            'is_marked': predicted_class_name == 'marked_bubble',
            'probabilities': {
                'marked_bubble': all_probabilities[0],
                'unmarked_bubble': all_probabilities[1]
            },
            'raw_probabilities': all_probabilities
        }
        
        if verbose:
            print(f"\n📊 Bubble Classification Result:")
            print(f"   Prediction: {predicted_class_name}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Is Marked: {'✅ Yes' if classification_result['is_marked'] else '❌ No'}")
            print(f"   Probabilities: Marked={all_probabilities[0]:.3f}, Unmarked={all_probabilities[1]:.3f}")
        
        return classification_result
    
    def classify_multiple_bubbles(self, image_paths, verbose=True):
        """
        Classify multiple bubble images
        
        Args:
            image_paths: List of image file paths
            verbose: Print results for each image
            
        Returns:
            list: List of classification results
        """
        
        if verbose:
            print(f"\n🔍 Processing {len(image_paths)} bubble images...")
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            if verbose:
                print(f"\n--- Image {i+1}/{len(image_paths)}: {Path(image_path).name} ---")
            
            try:
                result = self.classify_bubble(image_path, verbose=verbose)
                result['image_path'] = str(image_path)
                result['image_name'] = Path(image_path).name
                results.append(result)
                
            except Exception as e:
                if verbose:
                    print(f"❌ Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'image_name': Path(image_path).name,
                    'error': str(e),
                    'predicted_class': None,
                    'confidence': 0.0
                })
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_path': str(self.model_path),
            'model_task': self.model.task,
            'class_names': self.class_names,
            'input_size': '640x640 (automatic resize)',
            'supported_formats': ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            'description': 'YOLOv8n Classification model trained for OMR bubble detection'
        }

def demonstrate_usage():
    """Demonstrate various ways to use the bubble classifier"""
    
    print("=" * 60)
    print("🎓 OMR Bubble Classification Model - Usage Demonstration")
    print("=" * 60)
    
    # Model path
    model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please ensure the model has been trained and saved.")
        return
    
    # Initialize classifier
    try:
        classifier = OMRBubbleClassifier(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Show model information
    print("\n📋 Model Information:")
    model_info = classifier.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Demo 1: Single image classification
    print("\n" + "="*50)
    print("🔍 DEMO 1: Single Bubble Classification")
    print("="*50)
    
    # Find a sample image from training data
    sample_images_dir = Path("omr_training_results/trained_models/classification_dataset/train")
    sample_marked = list((sample_images_dir / "marked_bubble").glob("*.jpg"))
    sample_unmarked = list((sample_images_dir / "unmarked_bubble").glob("*.jpg"))
    
    if sample_marked:
        print(f"\n📸 Testing with marked bubble sample:")
        result = classifier.classify_bubble(sample_marked[0])
        
    if sample_unmarked:
        print(f"\n📸 Testing with unmarked bubble sample:")
        result = classifier.classify_bubble(sample_unmarked[0])
    
    # Demo 2: Batch classification
    print("\n" + "="*50)
    print("🔍 DEMO 2: Batch Classification")
    print("="*50)
    
    # Get a few sample images for batch processing
    sample_images = (sample_marked[:2] + sample_unmarked[:2]) if sample_marked and sample_unmarked else []
    
    if sample_images:
        results = classifier.classify_multiple_bubbles(sample_images, verbose=False)
        
        print(f"\n📊 Batch Results Summary:")
        print(f"{'Image Name':<30} {'Prediction':<15} {'Confidence':<12} {'Marked?'}")
        print("-" * 70)
        
        for result in results:
            if 'error' not in result:
                print(f"{result['image_name']:<30} {result['predicted_class']:<15} {result['confidence_percent']:<12} {'✅' if result['is_marked'] else '❌'}")
    
    # Demo 3: Using with OpenCV array
    print("\n" + "="*50)
    print("🔍 DEMO 3: Using with OpenCV Image Array")
    print("="*50)
    
    if sample_marked:
        # Load image as numpy array
        image_array = cv2.imread(str(sample_marked[0]))
        print(f"\n📸 Testing with OpenCV array (shape: {image_array.shape}):")
        result = classifier.classify_bubble(image_array)
    
    print("\n✅ All demonstrations completed!")
    print("\n💡 Usage Tips:")
    print("   1. Model automatically resizes images to 640x640")
    print("   2. Accepts various image formats (jpg, png, etc.)")
    print("   3. Returns confidence scores for both classes")
    print("   4. Best accuracy with clear, high-contrast bubble images")
    print("   5. Minimum recommended input size: 32x32 pixels")

def quick_test(image_path):
    """Quick test function for a single image"""
    
    model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return None
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    # Load and test
    classifier = OMRBubbleClassifier(model_path)
    result = classifier.classify_bubble(image_path)
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick test mode with provided image path
        image_path = sys.argv[1]
        print(f"🔍 Testing bubble classification on: {image_path}")
        result = quick_test(image_path)
        
        if result:
            print(f"\n📊 Result: {result['predicted_class']} ({result['confidence_percent']})")
    else:
        # Full demonstration mode
        demonstrate_usage()