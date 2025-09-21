#!/usr/bin/env python3
"""
Simple OMR Visualization
Shows what the model detects as marked vs unmarked on the main OMR image
"""

import cv2
import numpy as np
from pathlib import Path
from demo_bubble_classifier import OMRBubbleClassifier
import matplotlib.pyplot as plt

def detect_bubbles_simple(image):
    """Simple bubble detection using contours"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 2000:  # Filter by area
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.3:  # Filter by circularity
                    x, y, w, h = cv2.boundingRect(contour)
                    bubbles.append((x, y, w, h))
    
    return bubbles

def visualize_omr_simple(image_path, model_path, output_path="omr_visualization.jpg"):
    """Simple visualization of OMR detections"""
    print(f"🔍 Visualizing: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Load classifier
    classifier = OMRBubbleClassifier(model_path)
    
    # Detect bubbles
    bubbles = detect_bubbles_simple(image)
    print(f"✅ Found {len(bubbles)} potential bubbles")
    
    # Create visualization
    vis_image = image.copy()
    marked_count = 0
    unmarked_count = 0
    
    for i, (x, y, w, h) in enumerate(bubbles):
        # Extract bubble region
        bubble_roi = image[y:y+h, x:x+w]
        
        if bubble_roi.size > 0:
            # Classify the bubble
            result = classifier.classify_bubble(bubble_roi, verbose=False)
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            # Choose color and label
            if predicted_class == 'marked_bubble':
                color = (0, 255, 0)  # Green for marked
                label = f"MARKED ({confidence:.2f})"
                marked_count += 1
            else:
                color = (0, 0, 255)  # Red for unmarked  
                label = f"UNMARKED ({confidence:.2f})"
                unmarked_count += 1
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(vis_image, (x, y - label_size[1] - 5), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(vis_image, label, (x, y - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add summary
    summary = f"Total: {len(bubbles)} | Marked: {marked_count} | Unmarked: {unmarked_count}"
    cv2.putText(vis_image, summary, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis_image, summary, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    
    # Save result
    cv2.imwrite(output_path, vis_image)
    print(f"✅ Visualization saved to: {output_path}")
    print(f"📊 Results: {marked_count} marked, {unmarked_count} unmarked")
    
    return vis_image

def main():
    """Main function"""
    print("🎨 Simple OMR Visualization")
    print("=" * 50)
    
    # Test image and model
    test_image = 'dataset/Set A/Img1.jpeg'
    model_path = 'simple_training/omr_bubble_classifier/weights/best.pt'
    
    if Path(test_image).exists():
        result = visualize_omr_simple(test_image, model_path)
        if result is not None:
            print("✅ Visualization complete!")
            print("📁 Check 'omr_visualization.jpg' to see the bounding boxes")
        else:
            print("❌ Visualization failed")
    else:
        print(f"❌ Test image not found: {test_image}")

if __name__ == "__main__":
    main()
