# OMR Bubble Classification Model - Input Requirements & Usage Guide

## 📋 **Model Input Requirements**

### **Primary Input Format**
The trained YOLO model expects **cropped bubble images** with the following specifications:

#### **Image Specifications**
- **Format**: JPEG, PNG, or any OpenCV-compatible image format
- **Size**: The model automatically resizes to **640x640 pixels** during inference
- **Original Training Size**: 64x64 pixels (but model accepts any size)
- **Channels**: 3 (RGB color image)
- **Bit Depth**: 8-bit (0-255 pixel values)
- **Content**: Individual bubble regions cropped from OMR sheets

#### **Technical Details**
```python
Input Shape: (Height, Width, Channels)
Expected: (Any, Any, 3) → Automatically resized to (640, 640, 3)
Data Type: uint8 (0-255)
Color Space: RGB
```

---

## 🔧 **How to Use the Model**

### **Method 1: Direct File Path (Recommended)**
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt")

# Predict on a single bubble image
results = model("path/to/bubble_image.jpg")

# Get prediction
result = results[0]
predicted_class = result.names[result.probs.top1]
confidence = result.probs.top1conf.item()

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

### **Method 2: NumPy Array Input**
```python
import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("path/to/best.pt")

# Load image as numpy array
image = cv2.imread("bubble_image.jpg")  # Shape: (H, W, 3)

# Predict
results = model(image)
result = results[0]

# Get results
class_id = result.probs.top1
class_name = result.names[class_id]
confidence = result.probs.top1conf.item()
```

### **Method 3: Batch Processing**
```python
from ultralytics import YOLO
import glob

# Load model
model = YOLO("path/to/best.pt")

# Process multiple images
image_paths = glob.glob("bubble_images/*.jpg")
results = model(image_paths)

# Process results
for i, result in enumerate(results):
    class_name = result.names[result.probs.top1]
    confidence = result.probs.top1conf.item()
    print(f"Image {i+1}: {class_name} ({confidence:.2%})")
```

---

## 🎯 **Model Output Format**

### **Class Labels**
- **Class 0**: `marked_bubble` - Bubble is filled/marked
- **Class 1**: `unmarked_bubble` - Bubble is empty/unmarked

### **Prediction Results**
```python
# After running: results = model(image)
result = results[0]

# Available outputs:
result.probs.top1          # Predicted class ID (0 or 1)
result.probs.top1conf      # Confidence score (0.0 to 1.0)
result.probs.data          # Full probability tensor [prob_marked, prob_unmarked]
result.names               # Class names dictionary {0: 'marked_bubble', 1: 'unmarked_bubble'}
```

### **Example Output**
```
Input: bubble_crop.jpg
Prediction: marked_bubble
Confidence: 69.91%
Raw probabilities: [0.6991, 0.3009]
```

---

## 📐 **Input Preprocessing (Automatic)**

The model automatically handles:
1. **Resizing**: Any input size → 640×640 pixels
2. **Normalization**: Pixel values normalized to [0,1] range
3. **Format Conversion**: Various formats → RGB tensor
4. **Batch Preparation**: Single image → Batch format

**You don't need to manually preprocess images!**

---

## 🔍 **Input Requirements for Different Use Cases**

### **1. Individual Bubble Classification**
**Input**: Single cropped bubble image
```python
# Example bubble crop requirements:
- Minimum size: 32x32 pixels (recommended: 64x64 or larger)
- Content: Clear view of one bubble
- Background: Minimal noise around bubble
- Quality: Readable scan quality
```

### **2. Batch Processing OMR Sheets**
**Workflow**:
1. **Extract bubbles** from full OMR sheet using bubble detection
2. **Crop each bubble** to individual images
3. **Feed cropped bubbles** to the classification model
4. **Aggregate results** for full sheet analysis

### **3. Real-time Processing**
```python
# For video/camera input
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # Extract bubble region from frame
        bubble_crop = extract_bubble(frame)  # Your implementation
        
        # Classify bubble
        results = model(bubble_crop)
        prediction = results[0].names[results[0].probs.top1]
        
        # Display result
        cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('OMR Reader', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## ⚙️ **Model Configuration**

### **Inference Parameters**
```python
# Basic inference
results = model("image.jpg")

# Advanced inference with parameters
results = model(
    source="image.jpg",
    conf=0.5,          # Confidence threshold
    device="cpu",      # Device: "cpu" or "cuda"
    verbose=True,      # Print results
    save=False,        # Don't save prediction images
    show=False         # Don't display images
)
```

### **Performance Settings**
```python
# For faster inference (lower quality)
model.model.half()  # Use half precision

# For better accuracy (slower)
results = model("image.jpg", augment=True)  # Test-time augmentation
```

---

## 📊 **Quality Guidelines for Input Images**

### **Optimal Input Characteristics**
✅ **Good Quality**:
- Clear, high-contrast bubble boundaries
- Minimal noise and artifacts
- Proper lighting (not too dark/bright)
- Bubble centered in the crop
- Size: 64x64 pixels or larger

❌ **Poor Quality**:
- Blurry or low-resolution images
- Poor contrast between bubble and background
- Skewed or rotated bubbles
- Multiple bubbles in one crop
- Excessive noise or compression artifacts

### **Sample Quality Check**
```python
def check_image_quality(image_path):
    img = cv2.imread(image_path)
    
    # Basic quality checks
    height, width = img.shape[:2]
    
    print(f"Image size: {width}x{height}")
    print(f"Size check: {'✅ Good' if min(width, height) >= 32 else '❌ Too small'}")
    
    # Contrast check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    print(f"Contrast: {contrast:.1f} {'✅ Good' if contrast > 20 else '❌ Low contrast'}")
    
    return min(width, height) >= 32 and contrast > 20
```

---

## 🚀 **Quick Start Example**

```python
#!/usr/bin/env python3
"""
Quick start example for OMR bubble classification
"""

from ultralytics import YOLO
import cv2
import sys

def classify_bubble(image_path, model_path):
    """Classify a single bubble image"""
    
    # Load model
    model = YOLO(model_path)
    
    # Run prediction
    results = model(image_path)
    result = results[0]
    
    # Extract results
    class_name = result.names[result.probs.top1]
    confidence = result.probs.top1conf.item()
    
    return {
        'prediction': class_name,
        'confidence': confidence,
        'is_marked': class_name == 'marked_bubble',
        'raw_probabilities': result.probs.data.tolist()
    }

# Example usage
if __name__ == "__main__":
    model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
    image_path = "test_bubble.jpg"
    
    result = classify_bubble(image_path, model_path)
    
    print(f"Bubble Classification Result:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Is Marked: {result['is_marked']}")
```

---

## 🔧 **Integration with OMR Processing Pipeline**

### **Complete OMR Sheet Processing**
```python
def process_omr_sheet(sheet_image_path, model_path):
    """Process a complete OMR sheet"""
    
    # 1. Load the classification model
    model = YOLO(model_path)
    
    # 2. Load and preprocess the OMR sheet
    sheet_image = cv2.imread(sheet_image_path)
    
    # 3. Detect and extract all bubbles (your implementation)
    bubbles = detect_bubbles(sheet_image)  # Returns list of bubble crops
    
    # 4. Classify each bubble
    results = []
    for i, bubble_crop in enumerate(bubbles):
        prediction = model(bubble_crop)
        result = prediction[0]
        
        results.append({
            'bubble_id': i,
            'prediction': result.names[result.probs.top1],
            'confidence': result.probs.top1conf.item(),
            'is_marked': result.names[result.probs.top1] == 'marked_bubble'
        })
    
    return results
```

This comprehensive guide covers all the input requirements and usage patterns for the trained OMR bubble classification model. The model is flexible and handles various input formats automatically while providing reliable bubble classification results.