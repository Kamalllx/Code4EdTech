#!/usr/bin/env python3
"""
Show Individual Bubble Examples
Displays marked vs unmarked bubble examples side by side
"""

import cv2
import numpy as np
from pathlib import Path
from demo_bubble_classifier import OMRBubbleClassifier
import matplotlib.pyplot as plt

def show_bubble_examples():
    """Show examples of marked vs unmarked bubbles"""
    print("🔍 Showing Bubble Examples")
    print("=" * 50)
    
    # Load classifier
    classifier = OMRBubbleClassifier('simple_training/omr_bubble_classifier/weights/best.pt')
    
    # Get some marked bubble examples
    marked_path = Path('simple_dataset/val/marked_bubble')
    marked_files = list(marked_path.glob('*.jpg'))[:3]
    
    # Get some unmarked bubble examples  
    unmarked_path = Path('simple_dataset/val/unmarked_bubble')
    unmarked_files = list(unmarked_path.glob('*.jpg'))[:3]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Bubble Classification Examples', fontsize=16)
    
    # Show marked bubbles
    for i, img_file in enumerate(marked_files):
        if i < 3:
            img = cv2.imread(str(img_file))
            if img is not None:
                # Classify
                result = classifier.classify_bubble(str(img_file), verbose=False)
                predicted = result['predicted_class']
                confidence = result['confidence']
                
                # Display
                axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[0, i].set_title(f'Marked Bubble\nPredicted: {predicted}\nConfidence: {confidence:.2f}', 
                                   color='green' if predicted == 'marked_bubble' else 'red')
                axes[0, i].axis('off')
    
    # Show unmarked bubbles
    for i, img_file in enumerate(unmarked_files):
        if i < 3:
            img = cv2.imread(str(img_file))
            if img is not None:
                # Classify
                result = classifier.classify_bubble(str(img_file), verbose=False)
                predicted = result['predicted_class']
                confidence = result['confidence']
                
                # Display
                axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[1, i].set_title(f'Unmarked Bubble\nPredicted: {predicted}\nConfidence: {confidence:.2f}',
                                   color='green' if predicted == 'unmarked_bubble' else 'red')
                axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('bubble_examples.png', dpi=150, bbox_inches='tight')
    print("✅ Bubble examples saved as 'bubble_examples.png'")
    plt.show()

if __name__ == "__main__":
    show_bubble_examples()
