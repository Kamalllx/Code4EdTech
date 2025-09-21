# OMR Bubble Classification with YOLO

This repository contains a complete pipeline for training YOLO models to detect and classify marked/unmarked bubbles in OMR (Optical Mark Recognition) sheets, with comprehensive audit trail capabilities for teacher verification.

## Features

- **Automated Dataset Preparation**: Converts OMR images to YOLO training format
- **YOLO Model Training**: Supports both classification and detection approaches
- **Comprehensive Audit Trail**: JSON-based audit system for teacher verification
- **Performance Evaluation**: Detailed metrics, confusion matrices, and visualizations
- **Teacher Verification**: HTML forms and reports for educational oversight
- **Production Ready**: Error handling, logging, and quality assurance

## Quick Start

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python numpy pandas matplotlib seaborn scikit-learn pillow albumentations tqdm pyyaml
```

### 2. Prepare Your Data

Organize your OMR images in a directory:
```
raw_omr_images/
├── sheet1.jpg
├── sheet2.jpg
├── sheet3.png
└── ...
```

### 3. Train the Model

```bash
python train.py --data_dir "path/to/your/omr/images" --output_dir "training_results" --epochs 100
```

### 4. Review Results

After training, check the generated verification form:
- Open `training_results/audit_trail/verification/verification_form_YYYYMMDD_HHMMSS.html`
- Review model performance metrics
- Complete teacher verification checklist

## Command Line Options

```bash
python train.py \
    --data_dir "raw_omr_images" \           # Directory with OMR images
    --output_dir "omr_training_output" \    # Output directory
    --model_type "classification" \         # classification or detection
    --epochs 100 \                          # Training epochs
    --batch_size 16 \                       # Batch size
    --train_ratio 0.7 \                     # Training split ratio
    --val_ratio 0.2 \                       # Validation split ratio
    --test_ratio 0.1 \                      # Test split ratio
    --augment \                             # Enable data augmentation
    --skip_data_prep                        # Skip data preparation
```

## File Structure

```
OMR/
├── train.py                    # Main training script
├── data_preparation.py         # Dataset preparation module
├── yolo_trainer.py            # YOLO training pipeline
├── audit_system.py            # Audit trail and verification system
├── requirements.txt           # Python dependencies
└── README.md                  # This file

# Generated during training:
omr_training_output/
├── processed_dataset/         # YOLO format dataset
├── trained_models/           # Trained model files
├── audit_trail/             # Audit and verification files
│   ├── reports/             # JSON audit reports
│   ├── visualizations/      # Performance plots
│   ├── metrics/             # Evaluation metrics
│   ├── predictions/         # Model predictions
│   └── verification/        # Teacher verification forms
└── logs/                    # Training logs
```

## Model Architecture

The system supports two YOLO approaches:

1. **Classification Model (Recommended)**:
   - Crops individual bubbles from OMR sheets
   - Classifies each crop as "marked" or "unmarked"
   - More accurate for bubble classification

2. **Detection Model**:
   - Detects and classifies bubbles in full images
   - Single-pass inference
   - Better for end-to-end processing

## Audit Trail Features

### For Teachers/Administrators:
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Visual Reports**: Confusion matrices, confidence distributions
- **Sample Predictions**: Review actual model outputs
- **Verification Checklist**: Structured quality assurance
- **Error Analysis**: Detailed breakdown of model mistakes
- **HTML Reports**: Easy-to-read verification forms

### For Developers:
- **Complete Training Log**: All training parameters and results
- **Model Files**: Best and last model weights
- **Dataset Statistics**: Class distributions, augmentation details
- **Evaluation Metrics**: Comprehensive performance analysis
- **JSON Audit Trail**: Machine-readable audit records

## Quality Assurance

The system implements multiple quality checks:

- ✅ **Dataset Validation**: Ensures proper data structure
- ✅ **Performance Thresholds**: Accuracy ≥95% recommended
- ✅ **Error Rate Monitoring**: Tracks false positives/negatives
- ✅ **Confidence Calibration**: Analyzes prediction confidence
- ✅ **Teacher Verification**: Human oversight requirement
- ✅ **Audit Trail**: Complete traceability

## Example Usage

### Basic Training
```python
from data_preparation import OMRDataPreparator
from yolo_trainer import OMRYOLOTrainer
from audit_system import OMRAuditTrail

# Prepare dataset
preparator = OMRDataPreparator("raw_images", "yolo_dataset")
preparator.process_dataset()

# Train model
trainer = OMRYOLOTrainer("yolo_dataset", "models")
audit_report = trainer.run_complete_pipeline()

# Generate verification materials
audit = OMRAuditTrail("audit_trail")
verification_form = audit.create_teacher_verification_form(results)
```

### Custom Configuration
```python
# Custom training configuration
trainer.training_config.update({
    'epochs': 150,
    'batch_size': 32,
    'lr0': 0.001,
    'patience': 30
})

# Run training
results = trainer.train_model(use_classification=True)
```

## Performance Expectations

Based on typical OMR datasets:

- **Expected Accuracy**: 95-99%
- **Training Time**: 1-3 hours (depending on dataset size)
- **Inference Speed**: ~100 images/second
- **Model Size**: ~6MB (YOLOv8 nano)

## Troubleshooting

### Common Issues:

1. **No images found**: Check image formats (jpg, png, etc.)
2. **Low accuracy**: Increase training epochs or add more data
3. **Memory errors**: Reduce batch size
4. **Missing dependencies**: Install requirements.txt

### Performance Optimization:

1. **Data Quality**: Use high-resolution, well-lit images
2. **Augmentation**: Enable for small datasets
3. **Class Balance**: Ensure equal marked/unmarked samples
4. **Hyperparameters**: Tune learning rate and batch size

## Teacher Verification Process

1. **Review Performance Metrics**: Check accuracy, precision, recall
2. **Examine Sample Predictions**: Verify model outputs visually
3. **Analyze Error Patterns**: Understand model limitations
4. **Complete Checklist**: Systematic quality verification
5. **Provide Feedback**: Comments and recommendations
6. **Make Decision**: Approve, reject, or request revisions

## License

This project is designed for educational use at Innomatics Research Labs.

## Support

For questions or issues:
1. Check the audit trail logs for detailed error information
2. Review the troubleshooting section
3. Examine the verification reports for performance insights

## Contributing

When making improvements:
1. Maintain audit trail compatibility
2. Update verification forms as needed
3. Ensure teacher review process remains intact
4. Test with various OMR sheet formats