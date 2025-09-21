#!/usr/bin/env python3
"""
Enhanced OMR Processing System
Complete pipeline for processing OMR sheets with question number detection,
bubble detection, mapping, and database integration.
"""

import cv2
import numpy as np
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import time
import pytesseract
from PIL import Image

# Import our modules
from database import OMRDatabase
from demo_bubble_classifier import OMRBubbleClassifier

class EnhancedOMRProcessor:
    """Enhanced OMR processing system with question number detection"""
    
    def __init__(self, model_path: str, db_path: str = "omr_evaluation.db"):
        """Initialize enhanced OMR processor"""
        self.model_path = model_path
        self.db = OMRDatabase(db_path)
        
        # Load trained model
        try:
            self.classifier = OMRBubbleClassifier(model_path)
            logging.info(f"Enhanced OMR processor initialized with model: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Enhanced preprocessing for better detection"""
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
    
    def detect_question_numbers(self, image: np.ndarray) -> List[Dict]:
        """Detect question numbers using OCR and image processing"""
        # Convert back to BGR for OCR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Use OCR to detect text
        try:
            # Configure tesseract for better number detection
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            text_data = pytesseract.image_to_data(bgr_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            question_numbers = []
            
            for i in range(len(text_data['text'])):
                text = text_data['text'][i].strip()
                conf = int(text_data['conf'][i])
                
                # Filter for numbers with good confidence
                if text.isdigit() and conf > 50:
                    x = text_data['left'][i]
                    y = text_data['top'][i]
                    w = text_data['width'][i]
                    h = text_data['height'][i]
                    
                    question_numbers.append({
                        'number': int(text),
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'confidence': conf
                    })
            
            # Sort by position (top to bottom, left to right)
            question_numbers.sort(key=lambda q: (q['center'][1], q['center'][0]))
            
            logging.info(f"Detected {len(question_numbers)} question numbers")
            return question_numbers
            
        except Exception as e:
            logging.error(f"OCR error: {e}")
            return []
    
    def detect_bubbles_enhanced(self, image: np.ndarray) -> List[Dict]:
        """Enhanced bubble detection with better filtering"""
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Enhanced area filtering
            if 30 < area < 3000:  # More flexible area range
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # More lenient circularity check
                    if circularity > 0.3:  # Less strict circularity
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # More flexible aspect ratio
                        if 0.5 <= aspect_ratio <= 2.0:  # More lenient aspect ratio
                            bubbles.append({
                                'bbox': (x, y, w, h),
                                'area': area,
                                'circularity': circularity,
                                'center': (x + w//2, y + h//2),
                                'contour': contour
                            })
        
        # Sort bubbles by position
        bubbles.sort(key=lambda b: (b['center'][1], b['center'][0]))
        
        logging.info(f"Detected {len(bubbles)} potential bubbles")
        return bubbles
    
    def map_bubbles_to_questions(self, question_numbers: List[Dict], 
                                bubbles: List[Dict], 
                                questions_per_row: int = 5, 
                                options_per_question: int = 4) -> Dict:
        """Map bubbles to specific questions based on spatial relationships"""
        
        # Create question mapping
        question_mapping = {}
        
        # Group question numbers by rows
        question_rows = {}
        for q_num in question_numbers:
            row_y = q_num['center'][1]
            if row_y not in question_rows:
                question_rows[row_y] = []
            question_rows[row_y].append(q_num)
        
        # Sort rows by Y position
        sorted_rows = sorted(question_rows.items())
        
        current_question = 1
        
        for row_y, row_questions in sorted_rows:
            # Sort questions in this row by X position
            row_questions.sort(key=lambda q: q['center'][0])
            
            for q_num in row_questions:
                question_key = f"Q{q_num['number']}"
                
                # Find bubbles near this question number
                question_bubbles = []
                q_center = q_num['center']
                
                # Look for bubbles in the vicinity of this question
                for bubble in bubbles:
                    b_center = bubble['center']
                    
                    # Check if bubble is in the same row and to the right of question number
                    y_distance = abs(b_center[1] - q_center[1])
                    x_distance = b_center[0] - q_center[0]
                    
                    # Bubble should be in same row (within tolerance) and to the right
                    if y_distance < 50 and 20 < x_distance < 300:
                        question_bubbles.append(bubble)
                
                # Sort bubbles by X position (left to right)
                question_bubbles.sort(key=lambda b: b['center'][0])
                
                # Take only the first 4 bubbles (A, B, C, D options)
                question_bubbles = question_bubbles[:options_per_question]
                
                if len(question_bubbles) >= options_per_question:
                    question_mapping[question_key] = {
                        'question_number': q_num['number'],
                        'question_bbox': q_num['bbox'],
                        'bubbles': question_bubbles,
                        'options': ['A', 'B', 'C', 'D'][:len(question_bubbles)]
                    }
                
                current_question += 1
        
        logging.info(f"Mapped {len(question_mapping)} questions to bubbles")
        return question_mapping
    
    def extract_bubble_images(self, image_path: str, bubbles: List[Dict]) -> List[np.ndarray]:
        """Extract individual bubble images for classification"""
        image = cv2.imread(str(image_path))
        bubble_images = []
        
        for bubble in bubbles:
            x, y, w, h = bubble['bbox']
            # Add padding around bubble
            padding = 15
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
    
    def process_omr_sheet_enhanced(self, image_path: str, answer_key: Dict, 
                                  sheet_version: str = "Set A") -> Dict:
        """Enhanced OMR sheet processing with question number detection"""
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Detect question numbers
            question_numbers = self.detect_question_numbers(processed_image)
            
            # Detect bubbles
            bubbles = self.detect_bubbles_enhanced(processed_image)
            
            # Map bubbles to questions
            question_mapping = self.map_bubbles_to_questions(question_numbers, bubbles)
            
            # Extract all bubble images
            all_bubbles = []
            for q_data in question_mapping.values():
                all_bubbles.extend(q_data['bubbles'])
            
            bubble_images = self.extract_bubble_images(image_path, all_bubbles)
            
            # Classify all bubbles
            classifications = self.classify_bubbles(bubble_images)
            
            # Process answers with enhanced mapping
            student_answers = {}
            subject_scores = {}
            total_correct = 0
            
            # Initialize subject scores
            subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "English"]
            for subject in subjects:
                subject_scores[subject] = 0
            
            # Process each question
            for question_key, q_data in question_mapping.items():
                question_bubbles = q_data['bubbles']
                options = q_data['options']
                
                # Get classifications for this question's bubbles
                bubble_start_idx = len(student_answers) * len(options)
                q_classifications = classifications[bubble_start_idx:bubble_start_idx + len(options)]
                
                # Find marked option
                marked_option = None
                for i, classification in enumerate(q_classifications):
                    if classification['is_marked']:
                        marked_option = options[i]
                        break
                
                if marked_option is None:
                    marked_option = "UNMARKED"
                
                student_answers[question_key] = marked_option
                
                # Check against answer key
                correct_answer = answer_key.get(question_key, "UNKNOWN")
                is_correct = (marked_option == correct_answer)
                
                if is_correct:
                    total_correct += 1
                    # Determine subject based on question number
                    q_num = q_data['question_number']
                    subject_index = (q_num - 1) // 20  # 20 questions per subject
                    if subject_index < len(subjects):
                        subject_scores[subjects[subject_index]] += 1
            
            # Calculate total score and percentage
            total_questions = len(student_answers)
            percentage = (total_correct / total_questions * 100) if total_questions > 0 else 0
            
            processing_time = time.time() - start_time
            
            # Calculate model confidence
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
                'question_numbers_detected': len(question_numbers),
                'bubbles_detected': len(bubbles),
                'questions_mapped': len(question_mapping),
                'question_mapping': question_mapping,
                'timestamp': datetime.now().isoformat()
            }
            
            logging.info(f"Enhanced OMR processing completed: {total_correct}/{total_questions} correct ({percentage:.1f}%)")
            return result
            
        except Exception as e:
            logging.error(f"Error processing OMR sheet: {e}")
            raise
    
    def save_results_to_database(self, result: Dict, student_id: int, exam_id: int, 
                                image_path: str, sheet_version: str = "Set A") -> Dict:
        """Save processing results to database"""
        try:
            # Add OMR sheet to database
            omr_sheet_id = self.db.add_omr_sheet(student_id, exam_id, image_path, sheet_version)
            
            # Update processing status
            self.db.update_omr_processing(omr_sheet_id, image_path, 'completed')
            
            # Save evaluation result
            evaluation_id = self.db.add_evaluation_result(
                omr_sheet_id,
                result['subject_scores'],
                result['total_score'],
                result['percentage'],
                result['student_answers'],
                result['processing_time'],
                result['model_confidence']
            )
            
            # Add audit entry
            self.db.add_audit_entry(
                omr_sheet_id,
                'omr_processed_enhanced',
                {
                    'evaluation_id': evaluation_id,
                    'processing_time': result['processing_time'],
                    'model_confidence': result['model_confidence'],
                    'question_numbers_detected': result['question_numbers_detected'],
                    'bubbles_detected': result['bubbles_detected'],
                    'questions_mapped': result['questions_mapped']
                }
            )
            
            return {
                'omr_sheet_id': omr_sheet_id,
                'evaluation_id': evaluation_id,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Error saving to database: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_and_save(self, image_path: str, student_id: int, exam_id: int, 
                        answer_key: Dict, sheet_version: str = "Set A") -> Dict:
        """Complete processing pipeline: detect, map, classify, and save"""
        try:
            # Process OMR sheet
            result = self.process_omr_sheet_enhanced(image_path, answer_key, sheet_version)
            
            # Save to database
            db_result = self.save_results_to_database(result, student_id, exam_id, image_path, sheet_version)
            
            # Combine results
            final_result = {
                **result,
                'database_save': db_result,
                'image_path': image_path,
                'student_id': student_id,
                'exam_id': exam_id
            }
            
            return final_result
            
        except Exception as e:
            logging.error(f"Error in complete pipeline: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path,
                'student_id': student_id,
                'exam_id': exam_id
            }

# Example usage
if __name__ == "__main__":
    # Initialize enhanced processor
    model_path = "omr_training_results/trained_models/omr_bubble_classifier/weights/best.pt"
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Please train the model first using train.py")
        exit(1)
    
    processor = EnhancedOMRProcessor(model_path)
    
    # Example answer key
    answer_key = {
        "Q1": "A", "Q2": "B", "Q3": "C", "Q4": "D", "Q5": "A",
        "Q6": "B", "Q7": "C", "Q8": "D", "Q9": "A", "Q10": "B",
        # ... more answers
    }
    
    # Process a single sheet
    image_path = "path/to/omr_sheet.jpg"
    if Path(image_path).exists():
        result = processor.process_and_save(
            image_path, 
            student_id=1, 
            exam_id=1, 
            answer_key=answer_key
        )
        print(f"Processing result: {result}")
    else:
        print(f"Image not found: {image_path}")
