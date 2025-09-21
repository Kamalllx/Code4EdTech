"""
YOLO Service for OMR Flask Application
Handles YOLO model loading, inference, and bubble classification
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
import logging
from ultralytics import YOLO
import torch


class YOLOService:
    def __init__(self, model_path: str):
        """
        Initialize YOLO service
        
        Args:
            model_path: Path to trained YOLO model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_info = {}
        self.logger = logging.getLogger(__name__)
        
        # OMR-specific configuration
        self.omr_config = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'bubble_min_size': 10,
            'bubble_max_size': 50,
            'marking_threshold': 0.3,
            'expected_bubble_ratio': 0.7  # Expected width/height ratio for bubbles
        }
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load YOLO model"""
        try:
            if not self.model_path.exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load YOLO model
            self.model = YOLO(str(self.model_path))
            
            # Get model information
            self.model_info = {
                'model_path': str(self.model_path),
                'model_type': 'YOLOv8',
                'task': self.model.task,
                'input_size': self.model.model.args.get('imgsz', 640) if hasattr(self.model.model, 'args') else 640,
                'classes': getattr(self.model.model, 'names', {0: 'unmarked_bubble', 1: 'marked_bubble'}),
                'device': str(self.model.device),
                'loaded_successfully': True
            }
            
            self.logger.info(f"YOLO model loaded successfully: {self.model_info}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            self.model_info = {
                'model_path': str(self.model_path),
                'loaded_successfully': False,
                'error': str(e)
            }
            return False
    
    def check_model_status(self) -> Dict[str, Any]:
        """Check current model status"""
        return {
            'model_loaded': self.model is not None,
            'model_info': self.model_info,
            'torch_available': torch.cuda.is_available(),
            'device': str(self.model.device) if self.model else 'none'
        }
    
    def evaluate_omr_sheet(self, image: np.ndarray, answer_key: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate OMR sheet using YOLO model
        
        Args:
            image: Preprocessed OMR sheet image
            answer_key: Expected answers for scoring
            
        Returns:
            Evaluation results
        """
        if not self.model:
            return {
                'success': False,
                'error': 'YOLO model not loaded'
            }
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=self.omr_config['confidence_threshold'])
            
            # Parse detection results
            detections = self._parse_detections(results, image.shape)
            
            # Organize detections by rows/questions
            organized_detections = self._organize_detections_by_questions(detections, image.shape)
            
            # Extract student answers
            student_answers = self._extract_student_answers(organized_detections)
            
            # Calculate scores if answer key is provided
            scoring_results = {}
            if answer_key:
                scoring_results = self._calculate_scores(student_answers, answer_key)
            
            evaluation_results = {
                'success': True,
                'total_bubbles_detected': len(detections),
                'marked_bubbles': len([d for d in detections if d['class'] == 'marked_bubble']),
                'unmarked_bubbles': len([d for d in detections if d['class'] == 'unmarked_bubble']),
                'detections': detections,
                'organized_by_questions': organized_detections,
                'student_answers': student_answers,
                'scoring_results': scoring_results,
                'model_info': self.model_info,
                'processing_timestamp': self._get_timestamp()
            }
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in OMR evaluation: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _parse_detections(self, results, image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """Parse YOLO detection results"""
        detections = []
        
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                # Object detection format
                boxes = result.boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Convert bbox to center format
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection = {
                        'bbox': {
                            'x1': float(x1), 'y1': float(y1),
                            'x2': float(x2), 'y2': float(y2),
                            'center_x': float(center_x), 'center_y': float(center_y),
                            'width': float(width), 'height': float(height)
                        },
                        'confidence': confidence,
                        'class_id': class_id,
                        'class': self.model_info.get('classes', {}).get(class_id, f'class_{class_id}'),
                        'bubble_id': len(detections)
                    }
                    
                    # Validate detection as reasonable bubble
                    if self._is_valid_bubble_detection(detection, image_shape):
                        detections.append(detection)
            
            elif hasattr(result, 'probs') and result.probs is not None:
                # Classification format - need to detect bubbles first
                bubbles = self._detect_bubbles_cv(results[0].orig_img)
                for i, bubble_bbox in enumerate(bubbles):
                    # Classify each bubble
                    bubble_roi = self._extract_bubble_roi(results[0].orig_img, bubble_bbox)
                    classification_result = self.model(bubble_roi)
                    
                    if classification_result and len(classification_result) > 0:
                        probs = classification_result[0].probs
                        class_id = probs.top1
                        confidence = float(probs.top1conf)
                        
                        detection = {
                            'bbox': {
                                'x1': float(bubble_bbox[0]), 'y1': float(bubble_bbox[1]),
                                'x2': float(bubble_bbox[0] + bubble_bbox[2]), 
                                'y2': float(bubble_bbox[1] + bubble_bbox[3]),
                                'center_x': float(bubble_bbox[0] + bubble_bbox[2]/2),
                                'center_y': float(bubble_bbox[1] + bubble_bbox[3]/2),
                                'width': float(bubble_bbox[2]), 'height': float(bubble_bbox[3])
                            },
                            'confidence': confidence,
                            'class_id': class_id,
                            'class': self.model_info.get('classes', {}).get(class_id, f'class_{class_id}'),
                            'bubble_id': i
                        }
                        
                        if self._is_valid_bubble_detection(detection, image_shape):
                            detections.append(detection)
        
        return detections
    
    def _detect_bubbles_cv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect bubbles using computer vision as fallback"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use HoughCircles to detect circular bubbles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, 
            minRadius=self.omr_config['bubble_min_size'], 
            maxRadius=self.omr_config['bubble_max_size']
        )
        
        bubbles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Convert circle to bounding box
                x1, y1 = max(0, x - r), max(0, y - r)
                w, h = min(2*r, image.shape[1] - x1), min(2*r, image.shape[0] - y1)
                bubbles.append((x1, y1, w, h))
        
        return bubbles
    
    def _extract_bubble_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract bubble region of interest"""
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        # Resize to standard size for classification
        if roi.size > 0:
            roi = cv2.resize(roi, (64, 64))
        
        return roi
    
    def _is_valid_bubble_detection(self, detection: Dict[str, Any], image_shape: Tuple[int, int, int]) -> bool:
        """Validate if detection is a reasonable bubble"""
        bbox = detection['bbox']
        
        # Check size constraints
        width, height = bbox['width'], bbox['height']
        if width < self.omr_config['bubble_min_size'] or width > self.omr_config['bubble_max_size']:
            return False
        if height < self.omr_config['bubble_min_size'] or height > self.omr_config['bubble_max_size']:
            return False
        
        # Check aspect ratio (bubbles should be roughly circular)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        # Check if detection is within image bounds
        img_height, img_width = image_shape[:2]
        if (bbox['x1'] < 0 or bbox['y1'] < 0 or 
            bbox['x2'] > img_width or bbox['y2'] > img_height):
            return False
        
        # Check confidence threshold
        if detection['confidence'] < self.omr_config['confidence_threshold']:
            return False
        
        return True
    
    def _organize_detections_by_questions(self, detections: List[Dict[str, Any]], 
                                        image_shape: Tuple[int, int, int]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize detections by question rows"""
        if not detections:
            return {}
        
        # Sort detections by Y coordinate (top to bottom)
        sorted_detections = sorted(detections, key=lambda d: d['bbox']['center_y'])
        
        # Group by approximate Y coordinate (question rows)
        questions = {}
        current_question = 1
        current_y = sorted_detections[0]['bbox']['center_y']
        row_threshold = 30  # Pixels tolerance for same row
        
        current_row_detections = []
        
        for detection in sorted_detections:
            detection_y = detection['bbox']['center_y']
            
            # If Y coordinate is significantly different, start new question
            if abs(detection_y - current_y) > row_threshold:
                if current_row_detections:
                    # Sort current row by X coordinate (left to right)
                    current_row_detections.sort(key=lambda d: d['bbox']['center_x'])
                    questions[f'question_{current_question}'] = current_row_detections
                    current_question += 1
                
                current_row_detections = [detection]
                current_y = detection_y
            else:
                current_row_detections.append(detection)
        
        # Don't forget the last row
        if current_row_detections:
            current_row_detections.sort(key=lambda d: d['bbox']['center_x'])
            questions[f'question_{current_question}'] = current_row_detections
        
        return questions
    
    def _extract_student_answers(self, organized_detections: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Extract student answers from organized detections"""
        student_answers = {}
        
        for question_id, detections in organized_detections.items():
            marked_bubbles = [d for d in detections if d['class'] == 'marked_bubble']
            
            # Determine answer based on marked bubbles
            if not marked_bubbles:
                student_answers[question_id] = {
                    'answer': None,
                    'confidence': 0.0,
                    'multiple_marks': False,
                    'no_mark': True
                }
            elif len(marked_bubbles) == 1:
                # Single answer (normal case)
                bubble = marked_bubbles[0]
                # Determine answer letter based on position (A, B, C, D, etc.)
                bubble_index = next(i for i, d in enumerate(detections) if d == bubble)
                answer_letter = chr(ord('A') + bubble_index)
                
                student_answers[question_id] = {
                    'answer': answer_letter,
                    'confidence': bubble['confidence'],
                    'multiple_marks': False,
                    'no_mark': False,
                    'bubble_index': bubble_index
                }
            else:
                # Multiple marks detected
                best_bubble = max(marked_bubbles, key=lambda b: b['confidence'])
                bubble_index = next(i for i, d in enumerate(detections) if d == best_bubble)
                answer_letter = chr(ord('A') + bubble_index)
                
                student_answers[question_id] = {
                    'answer': answer_letter,
                    'confidence': best_bubble['confidence'],
                    'multiple_marks': True,
                    'no_mark': False,
                    'bubble_index': bubble_index,
                    'warning': f'Multiple marks detected ({len(marked_bubbles)} bubbles)'
                }
        
        return student_answers
    
    def _calculate_scores(self, student_answers: Dict[str, Any], 
                         answer_key: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate scores based on answer key"""
        scoring_results = {
            'total_questions': len(answer_key),
            'answered_questions': 0,
            'correct_answers': 0,
            'incorrect_answers': 0,
            'unanswered_questions': 0,
            'multiple_mark_penalties': 0,
            'detailed_results': {},
            'score_percentage': 0.0
        }
        
        for question_id, correct_answer in answer_key.items():
            student_answer_data = student_answers.get(question_id, {})
            student_answer = student_answer_data.get('answer')
            
            result = {
                'correct_answer': correct_answer,
                'student_answer': student_answer,
                'is_correct': False,
                'points': 0,
                'status': 'unanswered'
            }
            
            if student_answer is None:
                result['status'] = 'unanswered'
                scoring_results['unanswered_questions'] += 1
            elif student_answer_data.get('multiple_marks', False):
                result['status'] = 'multiple_marks'
                scoring_results['multiple_mark_penalties'] += 1
                # Depending on scoring rules, might give 0 points or negative points
            else:
                scoring_results['answered_questions'] += 1
                if student_answer.upper() == correct_answer.upper():
                    result['is_correct'] = True
                    result['points'] = 1
                    result['status'] = 'correct'
                    scoring_results['correct_answers'] += 1
                else:
                    result['status'] = 'incorrect'
                    scoring_results['incorrect_answers'] += 1
            
            result['confidence'] = student_answer_data.get('confidence', 0.0)
            scoring_results['detailed_results'][question_id] = result
        
        # Calculate percentage
        if scoring_results['total_questions'] > 0:
            scoring_results['score_percentage'] = (
                scoring_results['correct_answers'] / scoring_results['total_questions']
            ) * 100
        
        return scoring_results
    
    def create_annotated_image(self, image: np.ndarray, 
                             evaluation_results: Dict[str, Any]) -> np.ndarray:
        """Create annotated image with detection results"""
        annotated = image.copy()
        
        if not evaluation_results.get('success', False):
            return annotated
        
        detections = evaluation_results.get('detections', [])
        student_answers = evaluation_results.get('student_answers', {})
        scoring_results = evaluation_results.get('scoring_results', {})
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # Color based on class
            if detection['class'] == 'marked_bubble':
                color = (0, 255, 0)  # Green for marked
                thickness = 3
            else:
                color = (0, 0, 255)  # Red for unmarked
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence score
            conf_text = f"{detection['confidence']:.2f}"
            cv2.putText(annotated, conf_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add answer annotations
        organized_detections = evaluation_results.get('organized_by_questions', {})
        for question_id, question_detections in organized_detections.items():
            if question_detections:
                # Find the leftmost bubble in this question
                leftmost = min(question_detections, key=lambda d: d['bbox']['center_x'])
                x = int(leftmost['bbox']['x1'] - 50)
                y = int(leftmost['bbox']['center_y'])
                
                # Get student answer for this question
                answer_data = student_answers.get(question_id, {})
                answer = answer_data.get('answer', 'None')
                
                # Color based on correctness if scoring is available
                text_color = (255, 255, 255)  # White default
                if scoring_results and question_id in scoring_results.get('detailed_results', {}):
                    result = scoring_results['detailed_results'][question_id]
                    if result['status'] == 'correct':
                        text_color = (0, 255, 0)  # Green
                    elif result['status'] == 'incorrect':
                        text_color = (0, 0, 255)  # Red
                    elif result['status'] == 'multiple_marks':
                        text_color = (0, 165, 255)  # Orange
                
                # Draw question number and answer
                question_num = question_id.replace('question_', 'Q')
                text = f"{question_num}: {answer}"
                cv2.putText(annotated, text, (x, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Add summary information
        summary_y = 30
        summary_texts = [
            f"Total Bubbles: {evaluation_results.get('total_bubbles_detected', 0)}",
            f"Marked: {evaluation_results.get('marked_bubbles', 0)}",
            f"Unmarked: {evaluation_results.get('unmarked_bubbles', 0)}"
        ]
        
        if scoring_results:
            summary_texts.append(f"Score: {scoring_results.get('score_percentage', 0):.1f}%")
        
        for i, text in enumerate(summary_texts):
            cv2.putText(annotated, text, (10, summary_y + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def batch_evaluate_images(self, images: List[np.ndarray], 
                            answer_key: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Batch evaluate multiple OMR images"""
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.evaluate_omr_sheet(image, answer_key)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_index': i
                })
        
        return results