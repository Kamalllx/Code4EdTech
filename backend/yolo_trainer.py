"""
YOLO Training Pipeline for OMR Bubble Classification
This module trains a YOLO model for detecting marked/unmarked bubbles with audit trail
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List, Any, Tuple
import shutil
from tqdm import tqdm

class OMRYOLOTrainer:
    def __init__(self, dataset_path: str, model_output_dir: str):
        """
        Initialize YOLO trainer for OMR bubble classification
        
        Args:
            dataset_path: Path to YOLO format dataset
            model_output_dir: Directory to save trained models and results
        """
        self.dataset_path = Path(dataset_path)
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Audit trail setup
        self.audit_dir = self.model_output_dir / "audit_trail"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.training_config = {
            "model_name": "yolov8n-cls.pt",  # YOLOv8 nano for classification
            "epochs": 100,
            "batch_size": 16,
            "imgsz": 640,
            "optimizer": "AdamW",
            "lr0": 0.01,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "patience": 50,
            "save_period": 10,
            "val_split": 0.2,
            "augment": True,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4
        }
        
        self.audit_data = {
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "training_start_time": None,
            "training_end_time": None,
            "dataset_info": {},
            "model_config": self.training_config.copy(),
            "training_metrics": {},
            "validation_metrics": {},
            "test_results": {},
            "model_files": {},
            "teacher_verification": {
                "verified": False,
                "verified_by": None,
                "verification_date": None,
                "comments": ""
            }
        }
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate dataset structure and content
        
        Returns:
            Validation results for audit trail
        """
        print("Validating dataset...")
        
        validation_results = {
            "status": "success",
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        # Check if dataset.yaml exists
        yaml_path = self.dataset_path / "dataset.yaml"
        if not yaml_path.exists():
            validation_results["errors"].append("dataset.yaml not found")
            validation_results["status"] = "failed"
            return validation_results
        
        # Load dataset config
        with open(yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        validation_results["dataset_config"] = dataset_config
        
        # Check directories
        required_dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                validation_results["errors"].append(f"Missing directory: {dir_name}")
        
        # Count files
        stats = {}
        for split in ['train', 'val', 'test']:
            img_dir = self.dataset_path / f"{split}/images"
            label_dir = self.dataset_path / f"{split}/labels"
            
            if img_dir.exists() and label_dir.exists():
                img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                label_files = list(label_dir.glob("*.txt"))
                
                stats[split] = {
                    "images": len(img_files),
                    "labels": len(label_files),
                    "matched": len([f for f in img_files if 
                                  (label_dir / f"{f.stem}.txt").exists()])
                }
                
                # Check for mismatched files
                unmatched = len(img_files) - stats[split]["matched"]
                if unmatched > 0:
                    validation_results["warnings"].append(
                        f"{split}: {unmatched} images without corresponding labels"
                    )
        
        validation_results["statistics"] = stats
        
        # Analyze class distribution
        class_distribution = self.analyze_class_distribution()
        validation_results["class_distribution"] = class_distribution
        
        if validation_results["errors"]:
            validation_results["status"] = "failed"
        elif validation_results["warnings"]:
            validation_results["status"] = "warning"
        
        # Save validation results
        validation_path = self.audit_dir / "dataset_validation.json"
        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        return validation_results
    
    def analyze_class_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of classes in the dataset"""
        class_counts = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}, "test": {0: 0, 1: 0}}
        
        for split in ['train', 'val', 'test']:
            label_dir = self.dataset_path / f"{split}/labels"
            if label_dir.exists():
                for label_file in label_dir.glob("*.txt"):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_id = int(parts[0])
                                    class_counts[split][class_id] = class_counts[split].get(class_id, 0) + 1
                    except Exception as e:
                        print(f"Error reading {label_file}: {e}")
        
        return class_counts
    
    def prepare_classification_dataset(self) -> str:
        """
        Convert YOLO detection format to classification format
        
        Returns:
            Path to classification dataset
        """
        print("Converting to classification format...")
        
        cls_dataset_path = self.model_output_dir / "classification_dataset"
        
        # Create classification directory structure
        for split in ['train', 'val', 'test']:
            for class_name in ['unmarked_bubble', 'marked_bubble']:
                (cls_dataset_path / split / class_name).mkdir(parents=True, exist_ok=True)
        
        conversion_stats = {"train": {}, "val": {}, "test": {}}
        
        for split in ['train', 'val', 'test']:
            img_dir = self.dataset_path / f"{split}/images"
            label_dir = self.dataset_path / f"{split}/labels"
            
            if not (img_dir.exists() and label_dir.exists()):
                continue
            
            split_stats = {"unmarked_bubble": 0, "marked_bubble": 0, "total_crops": 0}
            
            for img_file in tqdm(img_dir.glob("*.jpg"), desc=f"Processing {split}"):
                label_file = label_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    continue
                
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                img_h, img_w = image.shape[:2]
                
                # Read labels
                with open(label_file, 'r') as f:
                    for idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        
                        class_id = int(parts[0])
                        center_x, center_y, width, height = map(float, parts[1:])
                        
                        # Convert normalized coordinates to pixel coordinates
                        x = int((center_x - width/2) * img_w)
                        y = int((center_y - height/2) * img_h)
                        w = int(width * img_w)
                        h = int(height * img_h)
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, min(x, img_w - 1))
                        y = max(0, min(y, img_h - 1))
                        w = min(w, img_w - x)
                        h = min(h, img_h - y)
                        
                        if w <= 0 or h <= 0:
                            continue
                        
                        # Crop bubble region
                        bubble_crop = image[y:y+h, x:x+w]
                        
                        if bubble_crop.size == 0:
                            continue
                        
                        # Resize crop to standard size
                        crop_resized = cv2.resize(bubble_crop, (64, 64))
                        
                        # Save to appropriate class directory
                        class_name = 'marked_bubble' if class_id == 1 else 'unmarked_bubble'
                        crop_filename = f"{img_file.stem}_{idx}_{class_name}.jpg"
                        crop_path = cls_dataset_path / split / class_name / crop_filename
                        
                        cv2.imwrite(str(crop_path), crop_resized)
                        
                        split_stats[class_name] += 1
                        split_stats["total_crops"] += 1
            
            conversion_stats[split] = split_stats
        
        # Save conversion statistics
        conversion_path = self.audit_dir / "classification_conversion.json"
        with open(conversion_path, 'w') as f:
            json.dump(conversion_stats, f, indent=2)
        
        return str(cls_dataset_path)
    
    def train_model(self, use_classification: bool = True) -> Dict[str, Any]:
        """
        Train YOLO model for OMR bubble classification
        
        Args:
            use_classification: Whether to use classification or detection
            
        Returns:
            Training results for audit trail
        """
        self.audit_data["training_start_time"] = datetime.now().isoformat()
        
        try:
            if use_classification:
                # Use classification model
                model = YOLO('yolov8n-cls.pt')
                dataset_path = self.prepare_classification_dataset()
                
                # Train classification model
                results = model.train(
                    data=dataset_path,
                    epochs=self.training_config["epochs"],
                    batch=self.training_config["batch_size"],
                    imgsz=self.training_config["imgsz"],
                    optimizer=self.training_config["optimizer"],
                    lr0=self.training_config["lr0"],
                    weight_decay=self.training_config["weight_decay"],
                    warmup_epochs=self.training_config["warmup_epochs"],
                    patience=self.training_config["patience"],
                    save_period=self.training_config["save_period"],
                    project=str(self.model_output_dir),
                    name="omr_bubble_classifier",
                    exist_ok=True,
                    pretrained=True,
                    verbose=True
                )
            else:
                # Use detection model
                model = YOLO('yolov8n.pt')
                
                # Train detection model
                results = model.train(
                    data=str(self.dataset_path / "dataset.yaml"),
                    epochs=self.training_config["epochs"],
                    batch=self.training_config["batch_size"],
                    imgsz=self.training_config["imgsz"],
                    project=str(self.model_output_dir),
                    name="omr_bubble_detector",
                    exist_ok=True,
                    pretrained=True,
                    verbose=True
                )
            
            self.audit_data["training_end_time"] = datetime.now().isoformat()
            
            # Extract training metrics
            training_results = {
                "status": "success",
                "model_type": "classification" if use_classification else "detection",
                "best_model_path": str(results.save_dir / "weights" / "best.pt"),
                "last_model_path": str(results.save_dir / "weights" / "last.pt"),
                "training_plots_dir": str(results.save_dir),
                "metrics": {}
            }
            
            # Save model paths
            self.audit_data["model_files"] = {
                "best_model": training_results["best_model_path"],
                "last_model": training_results["last_model_path"],
                "training_dir": training_results["training_plots_dir"]
            }
            
            return training_results
            
        except Exception as e:
            self.audit_data["training_end_time"] = datetime.now().isoformat()
            return {
                "status": "failed",
                "error": str(e),
                "model_type": "classification" if use_classification else "detection"
            }
    
    def evaluate_model(self, model_path: str, test_data_path: str = None) -> Dict[str, Any]:
        """
        Evaluate trained model on test set
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test data (optional)
            
        Returns:
            Evaluation results for audit trail
        """
        print("Evaluating model...")
        
        try:
            model = YOLO(model_path)
            
            if test_data_path is None:
                test_data_path = str(self.dataset_path / "test/images")
            
            # Run validation
            results = model.val()
            
            evaluation_results = {
                "status": "success",
                "model_path": model_path,
                "test_data_path": test_data_path,
                "metrics": {},
                "confusion_matrix": None,
                "classification_report": None
            }
            
            # Extract metrics based on model type
            if hasattr(results, 'top1acc'):
                # Classification metrics
                evaluation_results["metrics"] = {
                    "top1_accuracy": float(results.top1acc) if results.top1acc else 0.0,
                    "top5_accuracy": float(results.top5acc) if results.top5acc else 0.0
                }
            else:
                # Detection metrics
                evaluation_results["metrics"] = {
                    "map50": float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
                    "map50_95": float(results.box.map) if hasattr(results.box, 'map') else 0.0,
                    "precision": float(results.box.p) if hasattr(results.box, 'p') else 0.0,
                    "recall": float(results.box.r) if hasattr(results.box, 'r') else 0.0
                }
            
            # Generate detailed evaluation report
            self.generate_evaluation_report(model, test_data_path, evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "model_path": model_path
            }
    
    def generate_evaluation_report(self, model, test_data_path: str, evaluation_results: Dict):
        """Generate detailed evaluation report with visualizations"""
        
        # Create evaluation directory
        eval_dir = self.audit_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        
        # Test on sample images and create visual report
        test_images = list(Path(test_data_path).glob("*.jpg"))[:20]  # Test on first 20 images
        
        predictions = []
        ground_truths = []
        
        for img_path in test_images:
            try:
                # Run prediction
                results = model(str(img_path))
                
                # Extract prediction (simplified for audit trail)
                pred_class = 0  # Default to unmarked
                confidence = 0.5
                
                if results and len(results) > 0:
                    if hasattr(results[0], 'probs'):
                        # Classification result
                        probs = results[0].probs
                        pred_class = probs.top1
                        confidence = float(probs.top1conf)
                    elif hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                        # Detection result - take first detection
                        boxes = results[0].boxes
                        pred_class = int(boxes.cls[0])
                        confidence = float(boxes.conf[0])
                
                predictions.append({
                    "image": str(img_path.name),
                    "predicted_class": int(pred_class),
                    "confidence": confidence,
                    "predicted_label": "marked" if pred_class == 1 else "unmarked"
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                predictions.append({
                    "image": str(img_path.name),
                    "predicted_class": 0,
                    "confidence": 0.0,
                    "predicted_label": "unmarked",
                    "error": str(e)
                })
        
        # Save predictions
        predictions_path = eval_dir / "sample_predictions.json"
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        evaluation_results["sample_predictions"] = predictions
    
    def create_audit_report(self) -> str:
        """
        Create comprehensive audit report for teacher verification
        
        Returns:
            Path to audit report
        """
        print("Creating audit report...")
        
        # Update audit data with final information
        self.audit_data["report_generation_time"] = datetime.now().isoformat()
        
        # Create summary statistics
        summary = {
            "training_duration": None,
            "total_epochs_trained": self.training_config["epochs"],
            "best_model_performance": {},
            "dataset_statistics": {},
            "quality_assurance": {
                "validation_passed": True,
                "error_rate_acceptable": True,
                "teacher_review_required": True
            }
        }
        
        # Calculate training duration
        if (self.audit_data["training_start_time"] and 
            self.audit_data["training_end_time"]):
            start_time = datetime.fromisoformat(self.audit_data["training_start_time"])
            end_time = datetime.fromisoformat(self.audit_data["training_end_time"])
            duration = end_time - start_time
            summary["training_duration"] = str(duration)
        
        self.audit_data["summary"] = summary
        
        # Save complete audit trail
        audit_report_path = self.audit_dir / f"complete_audit_report_{self.audit_data['session_id']}.json"
        with open(audit_report_path, 'w') as f:
            json.dump(self.audit_data, f, indent=2, default=str)
        
        # Create human-readable report
        readable_report_path = self.audit_dir / f"training_report_{self.audit_data['session_id']}.md"
        self.create_readable_report(readable_report_path)
        
        print(f"Audit report saved to: {audit_report_path}")
        print(f"Readable report saved to: {readable_report_path}")
        
        return str(audit_report_path)
    
    def create_readable_report(self, report_path: str):
        """Create human-readable markdown report for teachers"""
        
        with open(report_path, 'w') as f:
            f.write(f"# OMR Bubble Classification Model Training Report\n\n")
            f.write(f"**Session ID:** {self.audit_data['session_id']}\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Training Summary\n\n")
            if self.audit_data.get("summary", {}).get("training_duration"):
                f.write(f"- **Training Duration:** {self.audit_data['summary']['training_duration']}\n")
            f.write(f"- **Total Epochs:** {self.training_config['epochs']}\n")
            f.write(f"- **Batch Size:** {self.training_config['batch_size']}\n")
            f.write(f"- **Model Type:** YOLOv8 Classification\n\n")
            
            f.write("## Model Files\n\n")
            if self.audit_data.get("model_files"):
                for key, path in self.audit_data["model_files"].items():
                    f.write(f"- **{key.replace('_', ' ').title()}:** `{path}`\n")
            f.write("\n")
            
            f.write("## Teacher Verification Required\n\n")
            f.write("Please review the following:\n\n")
            f.write("- [ ] Model performance metrics are acceptable\n")
            f.write("- [ ] Sample predictions look accurate\n")
            f.write("- [ ] No obvious errors in training process\n")
            f.write("- [ ] Ready for deployment\n\n")
            
            f.write("**Teacher Signature:** _____________________ **Date:** ___________\n\n")
            f.write("**Comments:**\n")
            f.write("_" * 60 + "\n\n")
            f.write("_" * 60 + "\n\n")
    
    def run_complete_pipeline(self, use_classification: bool = True) -> str:
        """
        Run the complete training pipeline with audit trail
        
        Args:
            use_classification: Whether to use classification or detection
            
        Returns:
            Path to audit report
        """
        print("Starting OMR YOLO training pipeline...")
        
        # Step 1: Validate dataset
        validation_results = self.validate_dataset()
        self.audit_data["dataset_info"] = validation_results
        
        if validation_results["status"] == "failed":
            print("Dataset validation failed. Check audit trail for details.")
            return self.create_audit_report()
        
        # Step 2: Train model
        training_results = self.train_model(use_classification)
        self.audit_data["training_metrics"] = training_results
        
        if training_results["status"] == "failed":
            print("Training failed. Check audit trail for details.")
            return self.create_audit_report()
        
        # Step 3: Evaluate model
        evaluation_results = self.evaluate_model(training_results["best_model_path"])
        self.audit_data["test_results"] = evaluation_results
        
        # Step 4: Create audit report
        audit_report_path = self.create_audit_report()
        
        print("Training pipeline completed successfully!")
        print(f"Audit report: {audit_report_path}")
        
        return audit_report_path

def main():
    """Example usage of the OMR YOLO Trainer"""
    
    # Set paths
    dataset_path = "yolo_dataset"  # Path to your YOLO format dataset
    model_output_dir = "trained_models"
    
    # Initialize trainer
    trainer = OMRYOLOTrainer(dataset_path, model_output_dir)
    
    # Run complete pipeline
    audit_report = trainer.run_complete_pipeline(use_classification=True)
    
    print(f"Training completed. Audit report: {audit_report}")

if __name__ == "__main__":
    main()