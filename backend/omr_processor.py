#!/usr/bin/env python3
"""
OMR Processing System
Complete pipeline for processing OMR sheets using trained YOLO models
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import time

# Import our modules
from database import OMRDatabase
from demo_bubble_classifier import OMRBubbleClassifier

class OMRProcessor:
    """Main OMR processing system"""
    
    def __init__(self, model_path: str, db_path: str = "omr_evaluation.db"):
        """Initialize OMR processor with trained model"""
        self.model_path = model_path
        self.db = OMRDatabase(db_path)
        
        # Load trained model
        try:
            self.classifier = OMRBubbleClassifier(model_path)
            logging.info(f"OMR processor initialized with model: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess OMR sheet image for better bubble detection"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return thresh
    
    def detect_bubble_grid(self, image: np.ndarray) -> List[Dict]:
        """Detect bubble grid structure in OMR sheet"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust these values based on your OMR sheet)
            if 50 < area < 2000:  # Typical bubble area range
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > 0.5:  # Circular enough to be a bubble
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        if 0.7 <= aspect_ratio <= 1.3:  # Roughly square
                            bubbles.append({
                                'bbox': (x, y, w, h),
                                'area': area,
                                'circularity': circularity,
                                'center': (x + w//2, y + h//2)
                            })
        
        return bubbles
    
    def group_bubbles_by_questions(self, bubbles: List[Dict], 
                                 questions_per_row: int = 5, 
                                 options_per_question: int = 4) -> Dict:
        """Group detected bubbles by questions"""
        # Sort bubbles by position (top to bottom, left to right)
        bubbles.sort(key=lambda b: (b['center'][1], b['center'][0]))
        
        questions = {}
        current_question = 1
        
        # Group bubbles into questions
        for i in range(0, len(bubbles), options_per_question):
            question_bubbles = bubbles[i:i+options_per_question]
            if len(question_bubbles) == options_per_question:
                questions[f"Q{current_question}"] = question_bubbles
                current_question += 1
        
        return questions
    
    def extract_bubble_images(self, image_path: str, bubbles: List[Dict]) -> List[np.ndarray]:
        """Extract individual bubble images for classification"""
        image = cv2.imread(str(image_path))
        bubble_images = []
        
        for bubble in bubbles:
            x, y, w, h = bubble['bbox']
            # Add padding around bubble
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            bubble_img = image[y1:y2, x1:x2]
            bubble_images.append(bubble_img)
        
        return bubble_images
    
    def classify_bubbles(self, bubble_images: List[np.ndarray]) -> List[Dict]:
        """Classify bubbles as marked or unmarked using trained model"""
        classifications = []
        
        for i, bubble_img in enumerate(bubble_images):
            try:
                result = self.classifier.classify_bubble(bubble_img, verbose=False)
                classifications.append({
                    'bubble_index': i,
                    'is_marked': result['is_marked'],
                    'confidence': result['confidence'],
                    'predicted_class': result['predicted_class']
                })
            except Exception as e:
                logging.error(f"Error classifying bubble {i}: {e}")
                classifications.append({
                    'bubble_index': i,
                    'is_marked': False,
                    'confidence': 0.0,
                    'predicted_class': 'error'
                })
        
        return classifications
    
    def process_omr_sheet(self, image_path: str, answer_key: Dict, 
                         sheet_version: str = "Set A") -> Dict:
        """Process a complete OMR sheet"""
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Detect bubbles
            bubbles = self.detect_bubble_grid(processed_image)
            logging.info(f"Detected {len(bubbles)} bubbles")
            
            # Group bubbles by questions
            questions = self.group_bubbles_by_questions(bubbles)
            logging.info(f"Grouped into {len(questions)} questions")
            
            # Extract bubble images
            bubble_images = self.extract_bubble_images(image_path, bubbles)
            
            # Classify bubbles
            classifications = self.classify_bubbles(bubble_images)
            
            # Process answers
            student_answers = []
            subject_scores = {}
            total_correct = 0
            
            # Initialize subject scores (assuming 5 subjects, 20 questions each)
            subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "English"]
            for subject in subjects:
                subject_scores[subject] = 0
            
            # Process each question
            for q_key, question_bubbles in questions.items():
                if len(question_bubbles) >= 4:  # Should have A, B, C, D options
                    # Get classifications for this question's bubbles
                    q_classifications = classifications[len(student_answers)*4:(len(student_answers)+1)*4]
                    
                    # Find marked option
                    marked_option = None
                    for i, classification in enumerate(q_classifications):
                        if classification['is_marked']:
                            marked_option = chr(65 + i)  # A, B, C, D
                            break
                    
                    if marked_option is None:
                        marked_option = "UNMARKED"
                    
                    student_answers.append(marked_option)
                    
                    # Check against answer key
                    correct_answer = answer_key.get(q_key, "UNKNOWN")
                    is_correct = (marked_option == correct_answer)
                    
                    if is_correct:
                        total_correct += 1
                        # Determine subject based on question number
                        q_num = int(q_key[1:])  # Extract question number
                        subject_index = (q_num - 1) // 20  # 20 questions per subject
                        if subject_index < len(subjects):
                            subject_scores[subjects[subject_index]] += 1
            
            # Calculate total score and percentage
            total_questions = len(student_answers)
            percentage = (total_correct / total_questions * 100) if total_questions > 0 else 0
            
            processing_time = time.time() - start_time
            
            # Calculate model confidence (average confidence of all classifications)
            confidences = [c['confidence'] for c in classifications if c['confidence'] > 0]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            result = {
                'student_answers': student_answers,
                'subject_scores': subject_scores,
                'total_score': total_correct,
                'total_questions': total_questions,
                'percentage': round(percentage, 2),
                'processing_time': round(processing_time, 2),
                'model_confidence': round(avg_confidence, 3),
                'sheet_version': sheet_version,
                'bubbles_detected': len(bubbles),
                'questions_processed': len(questions),
                'timestamp': datetime.now().isoformat()
            }
            
            logging.info(f"OMR processing completed: {total_correct}/{total_questions} correct ({percentage:.1f}%)")
            return result
            
        except Exception as e:
            logging.error(f"Error processing OMR sheet: {e}")
            raise
    
    def process_batch(self, image_paths: List[str], exam_id: int, 
                     answer_key: Dict, sheet_version: str = "Set A") -> List[Dict]:
        """Process multiple OMR sheets in batch"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logging.info(f"Processing sheet {i+1}/{len(image_paths)}: {Path(image_path).name}")
                
                # Process the sheet
                result = self.process_omr_sheet(image_path, answer_key, sheet_version)
                result['image_path'] = str(image_path)
                result['sheet_index'] = i
                
                # Save to database
                # Note: You'll need to add student_id and other metadata
                # This is a simplified version
                
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': str(image_path),
                    'error': str(e),
                    'sheet_index': i
                })
        
        return results
    
    def generate_report(self, results: List[Dict], output_path: str):
        """Generate comprehensive evaluation report"""
        report = {
            'summary': {
                'total_sheets': len(results),
                'successful_processing': len([r for r in results if 'error' not in r]),
                'failed_processing': len([r for r in results if 'error' in r]),
                'average_score': np.mean([r.get('percentage', 0) for r in results if 'error' not in r]),
                'processing_time': sum([r.get('processing_time', 0) for r in results if 'error' not in r])
            },
            'detailed_results': results,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Report generated: {output_path}")
        return report

# Example usage
if __name__ == "__main__":
    # Initialize processor
    model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Please train the model first using train.py")
        exit(1)
    
    processor = OMRProcessor(model_path)
    
    # Example answer key
    answer_key = {
        "Q1": "A", "Q2": "B", "Q3": "C", "Q4": "D", "Q5": "A",
        "Q6": "B", "Q7": "C", "Q8": "D", "Q9": "A", "Q10": "B",
        # ... more answers
    }
    
    # Process a single sheet
    image_path = "path/to/omr_sheet.jpg"
    if Path(image_path).exists():
        result = processor.process_omr_sheet(image_path, answer_key)
        print(f"Processing result: {result}")
    else:
        print(f"Image not found: {image_path}")