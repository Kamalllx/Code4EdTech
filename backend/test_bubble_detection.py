#!/usr/bin/env python3
"""
Test bubble detection on OMR images
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def test_bubble_detection(image_path):
    """Test bubble detection on a single image"""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Cannot load image: {image_path}")
        return
    
    print(f"Testing: {image_path}")
    print(f"Image shape: {image.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} contours")
    
    # Filter contours by area and circularity
    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Try different area ranges
        if 20 < area < 5000:  # Broader range
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.3:  # Lower threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 0.5 <= aspect_ratio <= 2.0:  # Broader aspect ratio
                        bubbles.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'circularity': circularity,
                            'center': (x + w//2, y + h//2)
                        })
    
    print(f"Found {len(bubbles)} potential bubbles")
    
    # Show results
    if bubbles:
        print("Bubble details:")
        for i, bubble in enumerate(bubbles[:5]):  # Show first 5
            print(f"  Bubble {i+1}: area={bubble['area']:.1f}, circularity={bubble['circularity']:.3f}, center={bubble['center']}")
    
    # Create visualization
    result_image = image.copy()
    for bubble in bubbles:
        x, y, w, h = bubble['bbox']
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, f"{bubble['area']:.0f}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save result
    output_path = f"test_result_{Path(image_path).stem}.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Result saved to: {output_path}")
    
    return len(bubbles)

def main():
    """Test bubble detection on multiple images"""
    
    # Test on a few images
    test_images = [
        "dataset/Set A/Img1.jpeg",
        "dataset/Set A/Img2.jpeg", 
        "dataset/Set A/Img20.jpeg",
        "dataset/Set B/Img9.jpeg",
        "dataset/Set B/Img21.jpeg"
    ]
    
    results = []
    for img_path in test_images:
        if Path(img_path).exists():
            bubble_count = test_bubble_detection(img_path)
            results.append((img_path, bubble_count))
        else:
            print(f"Image not found: {img_path}")
    
    print("\nSummary:")
    for img_path, count in results:
        print(f"{Path(img_path).name}: {count} bubbles")

if __name__ == "__main__":
    main()
