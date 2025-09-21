#!/usr/bin/env python3
"""
Visualize OMR Sheet Detections
Shows bounding boxes on the main OMR image for what the model detected as marked
"""

import cv2
import numpy as np
from pathlib import Path
from demo_bubble_classifier import OMRBubbleClassifier
import matplotlib.pyplot as plt
from omr_processor import OMRProcessor

def visualize_omr_detections(image_path, model_path, output_path=None):
    """
    Visualize detections on a full OMR sheet
    
    Args:
        image_path: Path to the OMR sheet image
        model_path: Path to the trained model
        output_path: Where to save the visualization (optional)
    """
    print(f"🔍 Visualizing detections on: {image_path}")
    
    # Load the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    try:
        # Load the processor
        processor = OMRProcessor(model_path)
        
        # Process the OMR sheet
        result = processor.process_omr_sheet(image, 'A')
        
        if not result.get('success', False):
            print("❌ Failed to process OMR sheet")
            return None
        
        bubbles = result.get('bubbles', [])
        print(f"✅ Detected {len(bubbles)} bubbles")
        
        # Draw bounding boxes and labels
        marked_count = 0
        unmarked_count = 0
        
        for i, bubble in enumerate(bubbles):
            x, y, w, h = bubble.get('bbox', [0, 0, 0, 0])
            is_filled = bubble.get('is_filled', False)
            confidence = bubble.get('confidence', 0)
            
            # Choose color based on detection
            if is_filled:
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
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(vis_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary text
        summary = f"Total: {len(bubbles)} | Marked: {marked_count} | Unmarked: {unmarked_count}"
        cv2.putText(vis_image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        # Save or display the result
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
            print(f"✅ Visualization saved to: {output_path}")
        else:
            # Display using matplotlib
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            plt.title(f"OMR Detections - {Path(image_path).name}")
            plt.axis('off')
            plt.show()
        
        print(f"📊 Results: {marked_count} marked, {unmarked_count} unmarked")
        return vis_image
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def test_multiple_images():
    """Test visualization on multiple OMR sheets"""
    print("🔍 Testing Visualization on Multiple OMR Sheets")
    print("=" * 60)
    
    model_path = 'simple_training/omr_bubble_classifier/weights/best.pt'
    
    # Test images
    test_images = [
        'dataset/Set A/Img1.jpeg',
        'dataset/Set A/Img2.jpeg', 
        'dataset/Set B/Img9.jpeg'
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n📸 Processing: {img_path}")
            
            # Create output path
            output_path = f"visualization_{Path(img_path).stem}.jpg"
            
            # Visualize
            result = visualize_omr_detections(img_path, model_path, output_path)
            
            if result is not None:
                print(f"✅ Visualization complete")
            else:
                print(f"❌ Failed to visualize")
        else:
            print(f"❌ Image not found: {img_path}")

def main():
    """Main function"""
    print("🎨 OMR Detection Visualization Tool")
    print("=" * 60)
    
    # Test on a single image first
    test_image = 'dataset/Set A/Img1.jpeg'
    model_path = 'simple_training/omr_bubble_classifier/weights/best.pt'
    
    if Path(test_image).exists():
        print(f"📸 Visualizing: {test_image}")
        result = visualize_omr_detections(test_image, model_path, "omr_detection_visualization.jpg")
        
        if result is not None:
            print("✅ Visualization created successfully!")
            print("📁 Check 'omr_detection_visualization.jpg' to see the results")
        else:
            print("❌ Visualization failed")
    else:
        print(f"❌ Test image not found: {test_image}")
    
    # Ask if user wants to test more images
    print("\n" + "=" * 60)
    print("🔍 Would you like to test more images?")
    print("Run: python visualize_omr_detections.py --all")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        test_multiple_images()
    else:
        main()
