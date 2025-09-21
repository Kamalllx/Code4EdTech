"""
Complete OMR Bubble Classification Training Pipeline
This is the main script that orchestrates the entire process from data preparation to model evaluation
with comprehensive audit trail for teacher verification.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging

# Import our custom modules
from data_preparation import OMRDataPreparator
from yolo_trainer import OMRYOLOTrainer
from audit_system import OMRAuditTrail

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging for the training pipeline"""
    
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def validate_input_data(data_dir: str) -> bool:
    """Validate that input data directory exists and contains images"""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        return False
    
    # Check for image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(data_path.glob(f"*{ext}"))
        image_files.extend(data_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"Error: No image files found in '{data_dir}'")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        return False
    
    print(f"Found {len(image_files)} images in input directory")
    return True

def main():
    """Main training pipeline"""
    
    parser = argparse.ArgumentParser(description='OMR Bubble Classification Training Pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing OMR images')
    parser.add_argument('--output_dir', type=str, default='omr_training_output',
                       help='Output directory for all results')
    parser.add_argument('--model_type', type=str, choices=['classification', 'detection'], 
                       default='classification', help='Type of YOLO model to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proportion of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Proportion of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Proportion of data for testing')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Apply data augmentation to training set')
    parser.add_argument('--skip_data_prep', action='store_true',
                       help='Skip data preparation (use existing processed data)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting OMR Bubble Classification Training Pipeline")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate input data
    if not validate_input_data(args.data_dir):
        logger.error("Input data validation failed")
        return 1
    
    try:
        # Initialize audit trail
        audit_dir = output_dir / "audit_trail"
        audit = OMRAuditTrail(str(audit_dir))
        logger.info(f"Audit trail initialized: {audit_dir}")
        
        # Step 1: Data Preparation
        if not args.skip_data_prep:
            logger.info("Step 1: Starting data preparation...")
            
            dataset_output_dir = output_dir / "processed_dataset"
            preparator = OMRDataPreparator(args.data_dir, str(dataset_output_dir))
            
            prep_results = preparator.process_dataset(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                augment_train=args.augment
            )
            
            logger.info("Data preparation completed")
            logger.info(f"Training images: {prep_results['dataset_summary']['train_images']}")
            logger.info(f"Validation images: {prep_results['dataset_summary']['val_images']}")
            logger.info(f"Test images: {prep_results['dataset_summary']['test_images']}")
            
        else:
            logger.info("Skipping data preparation (using existing data)")
            dataset_output_dir = output_dir / "processed_dataset"
            if not dataset_output_dir.exists():
                logger.error("Processed dataset directory not found. Cannot skip data preparation.")
                return 1
        
        # Step 2: Model Training
        logger.info("Step 2: Starting model training...")
        
        model_output_dir = output_dir / "trained_models"
        trainer = OMRYOLOTrainer(str(dataset_output_dir), str(model_output_dir))
        
        # Update training configuration
        trainer.training_config.update({
            'epochs': args.epochs,
            'batch_size': args.batch_size
        })
        
        # Run complete training pipeline
        audit_report_path = trainer.run_complete_pipeline(
            use_classification=(args.model_type == 'classification')
        )
        
        logger.info(f"Model training completed. Audit report: {audit_report_path}")
        
        # Step 3: Comprehensive Evaluation
        logger.info("Step 3: Starting comprehensive evaluation...")
        
        # Load training results from audit report
        with open(audit_report_path, 'r') as f:
            training_audit = json.load(f)
        
        # Get best model path
        best_model_path = training_audit.get('model_files', {}).get('best_model')
        if not best_model_path or not Path(best_model_path).exists():
            logger.error("Best model not found. Training may have failed.")
            return 1
        
        # Comprehensive evaluation
        test_data_path = dataset_output_dir / "test" / "images"
        eval_results = audit.evaluate_model_comprehensive(
            best_model_path, str(test_data_path)
        )
        
        logger.info("Model evaluation completed")
        if eval_results.get('metrics'):
            metrics = eval_results['metrics']
            logger.info(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
            logger.info(f"Precision: {metrics.get('precision', 0):.3f}")
            logger.info(f"Recall: {metrics.get('recall', 0):.3f}")
            logger.info(f"F1 Score: {metrics.get('f1_score', 0):.3f}")
        
        # Step 4: Generate Teacher Verification Materials
        logger.info("Step 4: Generating teacher verification materials...")
        
        # Create verification form
        verification_form_path = audit.create_teacher_verification_form(eval_results)
        logger.info(f"Teacher verification form created: {verification_form_path}")
        
        # Generate comprehensive report
        comprehensive_report_path = audit.generate_comprehensive_report(
            training_audit, eval_results
        )
        logger.info(f"Comprehensive audit report: {comprehensive_report_path}")
        
        # Step 5: Summary and Next Steps
        logger.info("Step 5: Training pipeline completed successfully!")
        
        print("\n" + "="*80)
        print("OMR BUBBLE CLASSIFICATION TRAINING COMPLETED")
        print("="*80)
        print(f"📁 Output Directory: {output_dir}")
        print(f"🤖 Best Model: {best_model_path}")
        print(f"📊 Audit Report: {audit_report_path}")
        print(f"✅ Verification Form: {verification_form_path}")
        
        if eval_results.get('metrics'):
            metrics = eval_results['metrics']
            print(f"\n📈 Model Performance:")
            print(f"   • Accuracy: {metrics.get('accuracy', 0)*100:.1f}%")
            print(f"   • Precision: {metrics.get('precision', 0):.3f}")
            print(f"   • Recall: {metrics.get('recall', 0):.3f}")
            print(f"   • F1 Score: {metrics.get('f1_score', 0):.3f}")
            
            accuracy = metrics.get('accuracy', 0)
            if accuracy >= 0.98:
                print(f"✅ Model performance is EXCELLENT - Ready for production!")
            elif accuracy >= 0.95:
                print(f"✅ Model performance is GOOD - Acceptable for production")
            elif accuracy >= 0.90:
                print(f"⚠️  Model performance is FAIR - Consider improvement")
            else:
                print(f"❌ Model performance is POOR - Needs significant improvement")
        
        print(f"\n🔍 Next Steps:")
        print(f"   1. Open the verification form in your browser: {verification_form_path.replace('.json', '.html')}")
        print(f"   2. Review model predictions and audit trail")
        print(f"   3. Complete teacher verification checklist")
        print(f"   4. If approved, deploy model for OMR evaluation")
        print("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)