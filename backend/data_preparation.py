"""
OMR Dataset Preparation Module for YOLO Training
This module handles loading, preprocessing, and preparing OMR sheet images for YOLO training
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
from typing import List, Tuple, Dict, Any
import albumentations as A
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

class OMRDataPreparator:
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize OMR Data Preparator
        
        Args:
            data_dir: Directory containing OMR images and annotations
            output_dir: Directory to save processed YOLO format data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.classes = ['unmarked_bubble', 'marked_bubble']  # 0: unmarked, 1: marked
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for YOLO format"""
        dirs = ['train/images', 'train/labels', 'val/images', 'val/labels', 'test/images', 'test/labels']
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess OMR sheet image
        
        Args:
            image_path: Path to the image
            
        Returns:
            Preprocessed image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale for better processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR for YOLO
        processed = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def detect_bubbles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect bubble regions in the image using computer vision
        
        Args:
            image: Input image
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (broader range)
            if 20 < area < 5000:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.3:  # Lower threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        if 0.5 <= aspect_ratio <= 2.0:  # Broader aspect ratio
                            bubbles.append((x, y, w, h))
        
        return bubbles
    
    def classify_bubble(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> int:
        """
        Classify if a bubble is marked or unmarked
        
        Args:
            image: Full image
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Class label (0: unmarked, 1: marked)
        """
        x, y, w, h = bbox
        bubble_roi = image[y:y+h, x:x+w]
        
        if bubble_roi.size == 0:
            return 0
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(bubble_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate darkness ratio
        total_pixels = gray_roi.shape[0] * gray_roi.shape[1]
        dark_pixels = np.sum(gray_roi < 128)
        darkness_ratio = dark_pixels / total_pixels
        
        # Threshold for marking detection (adjustable)
        return 1 if darkness_ratio > 0.3 else 0
    
    def convert_to_yolo_format(self, bbox: Tuple[int, int, int, int], 
                              img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert bounding box to YOLO format (normalized)
        
        Args:
            bbox: Bounding box (x, y, w, h)
            img_width: Image width
            img_height: Image height
            
        Returns:
            YOLO format (center_x, center_y, width, height) normalized
        """
        x, y, w, h = bbox
        center_x = (x + w/2) / img_width
        center_y = (y + h/2) / img_height
        norm_width = w / img_width
        norm_height = h / img_height
        
        return center_x, center_y, norm_width, norm_height
    
    def create_data_augmentation(self) -> A.Compose:
        """Create data augmentation pipeline"""
        return A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Blur(blur_limit=3, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.Affine(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def process_single_image(self, image_path: str, output_dir: str, 
                           augment: bool = False) -> Dict[str, Any]:
        """
        Process a single OMR image and extract bubbles
        
        Args:
            image_path: Path to image
            output_dir: Output directory
            augment: Whether to apply data augmentation
            
        Returns:
            Processing results for audit trail
        """
        try:
            # Load and preprocess image
            original_image = cv2.imread(image_path)
            processed_image = self.preprocess_image(image_path)
            
            # Detect bubbles
            bubbles = self.detect_bubbles(processed_image)
            
            if not bubbles:
                return {"status": "no_bubbles_found", "image": image_path}
            
            img_height, img_width = processed_image.shape[:2]
            
            # Create annotations
            annotations = []
            audit_data = {
                "image_path": image_path,
                "total_bubbles": len(bubbles),
                "marked_bubbles": 0,
                "unmarked_bubbles": 0,
                "bubbles": []
            }
            
            for i, bbox in enumerate(bubbles):
                # Classify bubble
                class_label = self.classify_bubble(processed_image, bbox)
                
                # Convert to YOLO format
                yolo_bbox = self.convert_to_yolo_format(bbox, img_width, img_height)
                
                annotations.append(f"{class_label} {' '.join(map(str, yolo_bbox))}")
                
                # Update audit data
                if class_label == 1:
                    audit_data["marked_bubbles"] += 1
                else:
                    audit_data["unmarked_bubbles"] += 1
                
                audit_data["bubbles"].append({
                    "bubble_id": i,
                    "bbox": bbox,
                    "yolo_bbox": yolo_bbox,
                    "class": "marked" if class_label == 1 else "unmarked",
                    "confidence": 0.8  # Placeholder confidence
                })
            
            # Save image and annotations
            image_name = Path(image_path).stem
            
            # Save processed image
            output_image_path = os.path.join(output_dir, "images", f"{image_name}.jpg")
            cv2.imwrite(output_image_path, processed_image)
            
            # Save annotations
            output_label_path = os.path.join(output_dir, "labels", f"{image_name}.txt")
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(annotations))
            
            # Apply data augmentation if requested
            if augment:
                self.apply_augmentation(processed_image, annotations, 
                                      output_dir, image_name, 3)
            
            audit_data["status"] = "success"
            return audit_data
            
        except Exception as e:
            return {"status": "error", "image": image_path, "error": str(e)}
    
    def apply_augmentation(self, image: np.ndarray, annotations: List[str], 
                          output_dir: str, base_name: str, num_augmentations: int = 3):
        """Apply data augmentation to increase dataset size"""
        transform = self.create_data_augmentation()
        
        # Parse YOLO annotations
        bboxes = []
        class_labels = []
        
        for ann in annotations:
            parts = ann.split()
            class_labels.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:]])
        
        for i in range(num_augmentations):
            try:
                # Apply augmentation
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Save augmented image
                aug_image_name = f"{base_name}_aug_{i}.jpg"
                aug_image_path = os.path.join(output_dir, "images", aug_image_name)
                cv2.imwrite(aug_image_path, augmented['image'])
                
                # Save augmented annotations
                aug_label_name = f"{base_name}_aug_{i}.txt"
                aug_label_path = os.path.join(output_dir, "labels", aug_label_name)
                
                with open(aug_label_path, 'w') as f:
                    for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                        f.write(f"{int(cls)} {' '.join(map(str, bbox))}\n")
                        
            except Exception as e:
                print(f"Augmentation failed for {base_name}: {e}")
    
    def process_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2, 
                       test_ratio: float = 0.1, augment_train: bool = True) -> Dict[str, Any]:
        """
        Process entire dataset and split into train/val/test
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            augment_train: Whether to augment training data
            
        Returns:
            Processing summary for audit trail
        """
        # Get all image files (including subdirectories)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            # Check direct files
            image_files.extend(self.data_dir.glob(f"*{ext}"))
            image_files.extend(self.data_dir.glob(f"*{ext.upper()}"))
            # Check subdirectories
            image_files.extend(self.data_dir.glob(f"**/*{ext}"))
            image_files.extend(self.data_dir.glob(f"**/*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No images found in {self.data_dir} or its subdirectories")
        
        print(f"Found {len(image_files)} images to process")
        
        # Split dataset
        train_files, temp_files = train_test_split(
            image_files, test_size=(1-train_ratio), random_state=42
        )
        
        val_files, test_files = train_test_split(
            temp_files, test_size=(test_ratio/(val_ratio + test_ratio)), random_state=42
        )
        
        # Process each split
        audit_trail = {
            "dataset_summary": {
                "total_images": len(image_files),
                "train_images": len(train_files),
                "val_images": len(val_files),
                "test_images": len(test_files)
            },
            "processing_results": {
                "train": [],
                "val": [],
                "test": []
            }
        }
        
        # Process training set
        print("Processing training set...")
        for img_path in tqdm(train_files, desc="Training"):
            result = self.process_single_image(
                str(img_path), 
                str(self.output_dir / "train"), 
                augment=augment_train
            )
            audit_trail["processing_results"]["train"].append(result)
        
        # Process validation set
        print("Processing validation set...")
        for img_path in tqdm(val_files, desc="Validation"):
            result = self.process_single_image(
                str(img_path), 
                str(self.output_dir / "val"), 
                augment=False
            )
            audit_trail["processing_results"]["val"].append(result)
        
        # Process test set
        print("Processing test set...")
        for img_path in tqdm(test_files, desc="Test"):
            result = self.process_single_image(
                str(img_path), 
                str(self.output_dir / "test"), 
                augment=False
            )
            audit_trail["processing_results"]["test"].append(result)
        
        # Create dataset.yaml for YOLO
        self.create_dataset_yaml()
        
        # Save audit trail
        audit_path = self.output_dir / "data_processing_audit.json"
        with open(audit_path, 'w') as f:
            json.dump(audit_trail, f, indent=2, default=str)
        
        print(f"Dataset processing complete. Audit trail saved to {audit_path}")
        return audit_trail
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset configuration saved to {yaml_path}")

def main():
    """Example usage of the OMR Data Preparator"""
    # Set paths
    data_dir = "raw_omr_images"  # Directory containing your OMR images
    output_dir = "yolo_dataset"
    
    # Initialize preparator
    preparator = OMRDataPreparator(data_dir, output_dir)
    
    # Process dataset
    audit_results = preparator.process_dataset(
        train_ratio=0.7,
        val_ratio=0.2, 
        test_ratio=0.1,
        augment_train=True
    )
    
    print("Data preparation completed!")
    print(f"Training images: {audit_results['dataset_summary']['train_images']}")
    print(f"Validation images: {audit_results['dataset_summary']['val_images']}")
    print(f"Test images: {audit_results['dataset_summary']['test_images']}")

if __name__ == "__main__":
    main()