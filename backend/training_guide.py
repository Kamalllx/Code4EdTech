"""
Step-by-Step Guide for Training OMR Model with Real Data
Follow this guide to train your YOLO model on actual OMR sheet data
"""

def print_guide():
    print("🎯 OMR Model Training - Step-by-Step Guide")
    print("="*70)
    
    print("\n📋 STEP 1: Download Your OMR Dataset")
    print("-"*50)
    print("1. Visit: https://drive.google.com/drive/folders/16MQMWcrtlNKdRIRd6sl_XlmAtowGa64a?usp=sharing")
    print("2. Download ALL OMR sheet images")
    print("3. Save them to: omr_dataset/ folder")
    print("4. Ensure images are in JPG/PNG format")
    
    print("\n📋 STEP 2: Verify Dataset")
    print("-"*50)
    print("Run: python setup_dataset.py")
    print("✅ Should show: 'Found X images in dataset!'")
    
    print("\n📋 STEP 3: Choose Training Mode")
    print("-"*50)
    print("🚀 FULL TRAINING (Recommended):")
    print("   python train.py --data_dir 'omr_dataset' --output_dir 'full_training' --epochs 100")
    print("")
    print("⚡ QUICK TEST (For testing setup):")
    print("   python train.py --data_dir 'omr_dataset' --output_dir 'quick_test' --epochs 20")
    
    print("\n📋 STEP 4: Monitor Training")
    print("-"*50)
    print("• Training will take 1-3 hours (depending on data size)")
    print("• Watch for accuracy improvements in the terminal")
    print("• Check logs in: [output_dir]/logs/")
    
    print("\n📋 STEP 5: Review Results")
    print("-"*50)
    print("After training completes:")
    print("1. Open: [output_dir]/audit_trail/verification/verification_form_*.html")
    print("2. Review model performance metrics")
    print("3. Check sample predictions")
    print("4. Complete teacher verification checklist")
    
    print("\n📋 STEP 6: Model Deployment")
    print("-"*50)
    print("If model passes verification (≥95% accuracy):")
    print("• Best model saved to: [output_dir]/trained_models/omr_bubble_classifier/weights/best.pt")
    print("• Use this model for production OMR evaluation")
    
    print("\n🔧 TROUBLESHOOTING")
    print("-"*50)
    print("❌ 'No training images found':")
    print("   → Check that OMR images contain visible bubbles")
    print("   → Ensure images are clear and high-quality")
    print("")
    print("❌ Low accuracy (<90%):")
    print("   → Add more training data")
    print("   → Increase epochs to 150-200")
    print("   → Check image quality")
    print("")
    print("❌ Memory errors:")
    print("   → Reduce batch size: --batch_size 8")
    print("   → Use smaller image size: --imgsz 416")
    
    print("\n📊 EXPECTED PERFORMANCE")
    print("-"*50)
    print("With good OMR data, expect:")
    print("• Accuracy: 95-99%")
    print("• Precision: >95%")
    print("• Recall: >95%")
    print("• F1-Score: >95%")
    
    print("\n🎓 TEACHER VERIFICATION CHECKLIST")
    print("-"*50)
    print("✅ Model accuracy ≥95%")
    print("✅ Sample predictions look correct")
    print("✅ Confusion matrix shows good performance")
    print("✅ No obvious biases in predictions")
    print("✅ Ready for production deployment")
    
    print("\n💡 NEXT STEPS")
    print("-"*50)
    print("1. Download your real OMR images")
    print("2. Run: python setup_dataset.py (to verify)")
    print("3. Run: python train.py --data_dir 'omr_dataset' --output_dir 'production_model'")
    print("4. Wait for training to complete")
    print("5. Review audit trail and verification form")
    print("6. Deploy if approved by teacher verification")

if __name__ == "__main__":
    print_guide()