"""
Dataset Setup Helper for OMR Training
This script helps you download and organize your OMR dataset for training
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil

def setup_data_directory():
    """Create the data directory structure"""
    data_dir = Path("omr_dataset")
    data_dir.mkdir(exist_ok=True)
    
    print(f"📁 Created data directory: {data_dir.absolute()}")
    return data_dir

def download_instructions():
    """Provide clear instructions for downloading the dataset"""
    print("\n🔗 Dataset Download Instructions:")
    print("="*60)
    print("1. Open this link in your browser:")
    print("   https://drive.google.com/drive/folders/16MQMWcrtlNKdRIRd6sl_XlmAtowGa64a?usp=sharing")
    print("\n2. Download all the OMR images:")
    print("   • Click on each image file")
    print("   • Download them one by one, OR")
    print("   • Select all files → Right-click → Download")
    print("\n3. Save all images to: omr_dataset/")
    print("   Expected file types: .jpg, .png, .jpeg, .bmp")
    print("\n4. Your folder should look like:")
    print("   omr_dataset/")
    print("   ├── sheet001.jpg")
    print("   ├── sheet002.png") 
    print("   ├── sheet003.jpg")
    print("   └── ...")
    print("\n✅ Once downloaded, run the training command!")

def check_dataset(data_dir):
    """Check if dataset is properly downloaded"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(data_dir.glob(f"*{ext}"))
        image_files.extend(data_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print("❌ No images found in omr_dataset/")
        print("Please download the images first using the instructions above.")
        return False
    
    print(f"✅ Found {len(image_files)} images in dataset!")
    print("Sample files:")
    for i, img in enumerate(image_files[:5]):
        print(f"   {i+1}. {img.name}")
    if len(image_files) > 5:
        print(f"   ... and {len(image_files) - 5} more")
    
    return True

def get_training_command():
    """Generate the training command"""
    print("\n🚀 Training Command:")
    print("="*60)
    
    # Basic training command
    command = """python train.py \\
    --data_dir "omr_dataset" \\
    --output_dir "training_results" \\
    --model_type "classification" \\
    --epochs 100 \\
    --batch_size 16 \\
    --augment"""
    
    print(command)
    
    print("\n⚡ Quick Test Command (for faster testing):")
    quick_command = """python train.py \\
    --data_dir "omr_dataset" \\
    --output_dir "quick_test" \\
    --epochs 20 \\
    --batch_size 8 \\
    --augment"""
    
    print(quick_command)

def main():
    """Main setup function"""
    print("🎯 OMR Dataset Setup Helper")
    print("="*60)
    
    # Create data directory
    data_dir = setup_data_directory()
    
    # Provide download instructions
    download_instructions()
    
    # Check if data already exists
    if check_dataset(data_dir):
        print("\n✨ Dataset is ready! You can proceed with training.")
        get_training_command()
    else:
        print("\n⏳ Please download the dataset first, then re-run this script to verify.")
    
    print("\n📋 What happens during training:")
    print("• Dataset preprocessing and augmentation")
    print("• YOLO model training (1-3 hours)")
    print("• Comprehensive evaluation and metrics")
    print("• Teacher verification form generation")
    print("• Audit trail for quality assurance")
    
    print(f"\n📂 All results will be saved to: training_results/")

if __name__ == "__main__":
    main()