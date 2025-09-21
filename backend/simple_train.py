#!/usr/bin/env python3
"""
Simplified OMR Training Script
Uses a smaller dataset and simpler approach
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

def create_simple_dataset():
    """Create a simplified dataset for training"""
    
    # Create directories
    dataset_dir = Path("simple_dataset")
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    for split in [train_dir, val_dir]:
        for class_name in ['marked_bubble', 'unmarked_bubble']:
            (split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process a few images to create a smaller dataset
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path("dataset").glob(f"**/*{ext}"))
    
    # Take only first 10 images for simplicity
    image_files = image_files[:10]
    
    print(f"Processing {len(image_files)} images...")
    
    marked_count = 0
    unmarked_count = 0
    
    for img_path in image_files:
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Simple bubble detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract bubbles
            bubbles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 50 < area < 2000:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity > 0.5:
                            x, y, w, h = cv2.boundingRect(contour)
                            bubbles.append((x, y, w, h))
            
            # Take only first 20 bubbles per image to limit dataset size
            bubbles = bubbles[:20]
            
            for i, (x, y, w, h) in enumerate(bubbles):
                # Extract bubble
                bubble_img = image[y:y+h, x:x+w]
                if bubble_img.size == 0:
                    continue
                
                # Simple classification: check if bubble is dark (marked)
                gray_bubble = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY)
                mean_intensity = np.mean(gray_bubble)
                
                # Classify as marked if dark, unmarked if light
                is_marked = mean_intensity < 100  # Threshold for marked/unmarked
                
                # Resize to standard size
                bubble_img = cv2.resize(bubble_img, (64, 64))
                
                # Save to appropriate directory
                if is_marked:
                    filename = f"{img_path.stem}_bubble_{i}_marked.jpg"
                    marked_count += 1
                else:
                    filename = f"{img_path.stem}_bubble_{i}_unmarked.jpg"
                    unmarked_count += 1
                
                # Split into train/val
                if i % 4 == 0:  # Every 4th bubble goes to validation
                    save_dir = val_dir
                else:
                    save_dir = train_dir
                
                cv2.imwrite(str(save_dir / ('marked_bubble' if is_marked else 'unmarked_bubble') / filename), bubble_img)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Created dataset with {marked_count} marked and {unmarked_count} unmarked bubbles")
    return dataset_dir

def train_simple_model():
    """Train a simple classification model"""
    
    print("Creating simplified dataset...")
    dataset_dir = create_simple_dataset()
    
    print("Starting training...")
    
    # Load YOLO model
    model = YOLO('yolov8n-cls.pt')
    
    # Train the model
    results = model.train(
        data=str(dataset_dir),
        epochs=20,
        batch=8,
        imgsz=64,
        device='cpu',  # Use CPU to avoid memory issues
        project='simple_training',
        name='omr_bubble_classifier',
        exist_ok=True
    )
    
    print("Training completed!")
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    
    return results.save_dir / "weights" / "best.pt"

if __name__ == "__main__":
    try:
        model_path = train_simple_model()
        print(f"✅ Training successful! Model saved to: {model_path}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
