"""
Create realistic OMR sheet images for training
This script generates synthetic OMR sheets with marked and unmarked bubbles
"""

import cv2
import numpy as np
from pathlib import Path
import random

def create_realistic_omr_sheet(sheet_id, marked_probability=0.3):
    """
    Create a realistic OMR sheet with bubbles
    
    Args:
        sheet_id: Unique identifier for the sheet
        marked_probability: Probability of a bubble being marked
    
    Returns:
        numpy array representing the OMR sheet image
    """
    # Create white background (A4 size scaled down)
    height, width = 1200, 900
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some texture/noise to make it more realistic
    noise = np.random.normal(0, 5, (height, width, 3)).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Add header text
    cv2.putText(image, "INNOMATICS RESEARCH LABS", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(image, "OMR EVALUATION SHEET", (50, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, f"Sheet ID: {sheet_id:03d}", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Create question grid (20 questions with 4 options each)
    start_x, start_y = 100, 200
    bubble_radius = 12
    option_spacing = 60
    question_spacing = 50
    
    marked_bubbles = []
    unmarked_bubbles = []
    
    for question in range(20):
        # Question label
        q_y = start_y + question * question_spacing
        cv2.putText(image, f"Q{question+1:02d}:", (50, q_y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Options A, B, C, D
        for option in range(4):
            option_x = start_x + option * option_spacing
            option_y = q_y
            
            # Draw option label
            option_label = chr(65 + option)  # A, B, C, D
            cv2.putText(image, option_label, (option_x - 25, option_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Decide if this bubble should be marked
            is_marked = random.random() < marked_probability
            
            if is_marked:
                # Draw filled circle (marked bubble)
                cv2.circle(image, (option_x, option_y), bubble_radius, (0, 0, 0), -1)
                # Add some variation in marking intensity
                overlay = np.zeros_like(image)
                cv2.circle(overlay, (option_x, option_y), bubble_radius-2, (50, 50, 50), -1)
                image = cv2.addWeighted(image, 0.8, overlay, 0.2, 0)
                
                marked_bubbles.append({
                    'center': (option_x, option_y),
                    'radius': bubble_radius,
                    'question': question + 1,
                    'option': option_label
                })
            else:
                # Draw empty circle (unmarked bubble)
                cv2.circle(image, (option_x, option_y), bubble_radius, (0, 0, 0), 2)
                
                unmarked_bubbles.append({
                    'center': (option_x, option_y),
                    'radius': bubble_radius,
                    'question': question + 1,
                    'option': option_label
                })
    
    # Add some scanning artifacts and imperfections
    # Slight skew
    angle = random.uniform(-2, 2)
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    # Add slight blur (camera focus issue)
    if random.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Add brightness variation
    brightness_factor = random.uniform(0.85, 1.15)
    image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    
    return image, marked_bubbles, unmarked_bubbles

def create_omr_dataset(num_sheets=50):
    """Create a complete OMR dataset"""
    
    # Clear existing dataset
    data_dir = Path("omr_dataset")
    if data_dir.exists():
        for file in data_dir.glob("*"):
            if file.is_file():
                file.unlink()
    
    data_dir.mkdir(exist_ok=True)
    
    print(f"Creating {num_sheets} realistic OMR sheets...")
    
    dataset_info = {
        'total_sheets': num_sheets,
        'total_marked_bubbles': 0,
        'total_unmarked_bubbles': 0,
        'sheets': []
    }
    
    for i in range(num_sheets):
        # Vary marking probability for different students
        marking_prob = random.uniform(0.15, 0.45)  # 15-45% questions answered
        
        image, marked, unmarked = create_realistic_omr_sheet(i + 1, marking_prob)
        
        # Save image
        filename = f"omr_sheet_{i+1:03d}.jpg"
        filepath = data_dir / filename
        cv2.imwrite(str(filepath), image)
        
        # Update statistics
        dataset_info['total_marked_bubbles'] += len(marked)
        dataset_info['total_unmarked_bubbles'] += len(unmarked)
        dataset_info['sheets'].append({
            'filename': filename,
            'marked_bubbles': len(marked),
            'unmarked_bubbles': len(unmarked),
            'marking_rate': len(marked) / (len(marked) + len(unmarked))
        })
        
        if (i + 1) % 10 == 0:
            print(f"Created {i + 1}/{num_sheets} sheets...")
    
    print(f"\n✅ Dataset creation complete!")
    print(f"📊 Dataset Statistics:")
    print(f"   • Total sheets: {dataset_info['total_sheets']}")
    print(f"   • Total marked bubbles: {dataset_info['total_marked_bubbles']}")
    print(f"   • Total unmarked bubbles: {dataset_info['total_unmarked_bubbles']}")
    print(f"   • Average marking rate: {dataset_info['total_marked_bubbles']/(dataset_info['total_marked_bubbles']+dataset_info['total_unmarked_bubbles'])*100:.1f}%")
    print(f"   • Images saved to: {data_dir.absolute()}")
    
    # Save dataset info
    import json
    with open(data_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    return dataset_info

if __name__ == "__main__":
    # Create dataset
    dataset_info = create_omr_dataset(50)
    print(f"\n🚀 Ready for training!")
    print(f"Run: python train.py --data_dir 'omr_dataset' --output_dir 'training_results'")