"""
Example usage script for OMR Bubble Classification
This script demonstrates how to use the OMR training pipeline
"""

import os
from pathlib import Path

def download_sample_data():
    """
    Instructions for downloading the provided Google Drive dataset
    """
    print("📥 Dataset Download Instructions:")
    print("="*50)
    print("1. Visit: https://drive.google.com/drive/folders/16MQMWcrtlNKdRIRd6sl_XlmAtowGa64a?usp=sharing")
    print("2. Download all images to a local folder (e.g., 'sample_data')")
    print("3. Organize images in the following structure:")
    print("   sample_data/")
    print("   ├── image1.jpg")
    print("   ├── image2.jpg")
    print("   └── ...")
    print("4. Run the training command below")
    print()

def run_training_example():
    """
    Example training command
    """
    print("🚀 Training Command Example:")
    print("="*50)
    
    # Example command for the provided dataset
    command = """python train.py \\
    --data_dir "sample_data" \\
    --output_dir "omr_results" \\
    --model_type "classification" \\
    --epochs 100 \\
    --batch_size 16 \\
    --augment"""
    
    print(command)
    print()
    print("📊 Expected Output:")
    print("- Processed dataset in YOLO format")
    print("- Trained YOLO model (.pt files)")
    print("- Comprehensive audit trail")
    print("- Teacher verification form (HTML)")
    print("- Performance visualizations")
    print()

def show_verification_process():
    """
    Explain the teacher verification process
    """
    print("✅ Teacher Verification Process:")
    print("="*50)
    print("1. Training completes and generates audit trail")
    print("2. Open verification form: omr_results/audit_trail/verification/verification_form_*.html")
    print("3. Review model performance metrics:")
    print("   • Accuracy should be ≥95%")
    print("   • Check precision and recall")
    print("   • Examine confusion matrix")
    print("4. Verify sample predictions visually")
    print("5. Complete verification checklist")
    print("6. Provide teacher feedback and approval")
    print()

def quick_test_example():
    """
    Quick test with minimal data
    """
    print("🧪 Quick Test Example (for testing setup):")
    print("="*50)
    
    test_command = """# Quick test with 10 epochs
python train.py \\
    --data_dir "sample_data" \\
    --output_dir "quick_test" \\
    --epochs 10 \\
    --batch_size 8"""
    
    print(test_command)
    print()
    print("This will run a quick training to verify everything works correctly.")
    print()

def main():
    """
    Main example runner
    """
    print("🎯 OMR Bubble Classification - Example Usage")
    print("="*70)
    print()
    
    download_sample_data()
    run_training_example()
    show_verification_process()
    quick_test_example()
    
    print("📚 Additional Information:")
    print("="*50)
    print("• Full documentation: README.md")
    print("• Training logs: omr_results/logs/")
    print("• Model files: omr_results/trained_models/")
    print("• Audit trail: omr_results/audit_trail/")
    print()
    print("🆘 If you encounter issues:")
    print("• Check requirements.txt for dependencies")
    print("• Verify image formats (jpg, png supported)")
    print("• Review logs for detailed error messages")
    print("• Ensure sufficient disk space (>2GB recommended)")

if __name__ == "__main__":
    main()