"""
YOLO Service for OMR Flask Application
Handles YOLO model loading, inference, and bubble classification
Updated to work with classification-based bubble detection from backend training pipeline
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json
import logging
from ultralytics import YOLO
import torch
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean


class YOLOService:
    def __init__(self, model_path: str):
        """
        Initialize YOLO service for classification-based bubble detection
        
        Args:
            model_path: Path to trained YOLO classification model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_info = {}
        self.logger = logging.getLogger(__name__)
        
        # OMR-specific configuration aligned with backend training
        self.omr_config = {
            'confidence_threshold': 0.5,  # Minimum confidence for bubble classification
            'bubble_detection_threshold': 0.3,  # For initial bubble detection
            'bubble_min_size': 15,  # Minimum bubble size in pixels
            'bubble_max_size': 80,  # Maximum bubble size in pixels
            'marking_threshold': 0.5,  # Threshold for marked vs unmarked classification
            'expected_bubble_ratio': 0.8,  # Expected width/height ratio for bubbles
            'grid_tolerance': 20,  # Pixel tolerance for grid alignment
            'crop_padding': 5  # Padding around bubble crops
        }
        
        # Class names from backend training
        self.class_names = {
            0: 'marked_bubble',
            1: 'unmarked_bubble'
        }
        
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load YOLO classification model"""
        try:
            if not self.model_path.exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load YOLO classification model
            self.model = YOLO(str(self.model_path))
            
            # Verify model type (should be classification)
            if hasattr(self.model.model, 'task') and self.model.model.task != 'classify':
                self.logger.warning(f"Model task is {self.model.model.task}, expected 'classify'")
            
            # Get model information
            self.model_info = {
                'model_path': str(self.model_path),
                'model_type': 'YOLOv8-Classification',
                'task': getattr(self.model, 'task', 'classify'),
                'input_size': 640,  # Standard classification input size
                'classes': self.class_names,
                'num_classes': len(self.class_names),
                'device': str(self.model.device) if hasattr(self.model, 'device') else 'cpu',
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
        Evaluate OMR sheet using classification-based bubble detection
        
        Args:
            image: Preprocessed OMR sheet image
            answer_key: Expected answers for scoring
            
        Returns:
            Evaluation results with bubble classifications
        """
        if not self.model:
            return {
                'success': False,
                'error': 'YOLO classification model not loaded'
            }
        
        try:
            # Step 1: Detect bubble regions using computer vision
            bubble_regions = self._detect_bubble_regions(image)
            
            if not bubble_regions:
                return {
                    'success': False,
                    'error': 'No bubble regions detected in the image'
                }
            
            # Step 2: Classify each bubble using YOLO
            classified_bubbles = []
            for i, region in enumerate(bubble_regions):
                bubble_crop = self._crop_bubble(image, region)
                classification = self._classify_bubble(bubble_crop)
                
                classification.update({
                    'bubble_id': i,
                    'region': region,
                    'position': {
                        'x': region['center_x'],
                        'y': region['center_y']
                    }
                })
                classified_bubbles.append(classification)
            
            # Step 3: Organize bubbles by questions/rows
            organized_bubbles = self._organize_bubbles_by_questions(classified_bubbles, image.shape)
            
            # Step 4: Extract student answers
            student_answers = self._extract_student_answers_classification(organized_bubbles)
            
            # Step 5: Calculate scores if answer key is provided
            scoring_results = {}
            if answer_key:
                scoring_results = self._calculate_scores(student_answers, answer_key)
            
            # Count marked/unmarked bubbles
            marked_count = len([b for b in classified_bubbles if b['is_marked']])
            unmarked_count = len(classified_bubbles) - marked_count
            
            evaluation_results = {
                'success': True,
                'total_bubbles_detected': len(classified_bubbles),
                'marked_bubbles': marked_count,
                'unmarked_bubbles': unmarked_count,
                'bubble_classifications': classified_bubbles,
                'organized_by_questions': organized_bubbles,
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
                'error': str(e),
                'details': str(e)
            }
    
    def _detect_bubble_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect bubble regions using computer vision techniques
        
        Args:
            image: Input OMR sheet image
            
        Returns:
            List of bubble region coordinates
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bubble_regions = []
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                
                # Filter by area
                min_area = self.omr_config['bubble_min_size'] ** 2
                max_area = self.omr_config['bubble_max_size'] ** 2
                
                if min_area <= area <= max_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (should be roughly circular)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 <= aspect_ratio <= 2.0:  # Allow some tolerance
                        
                        # Calculate center and dimensions
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        bubble_regions.append({
                            'x': x, 'y': y, 'width': w, 'height': h,
                            'center_x': center_x, 'center_y': center_y,
                            'area': area, 'aspect_ratio': aspect_ratio,
                            'contour': contour
                        })
            
            self.logger.info(f"Detected {len(bubble_regions)} potential bubble regions")
            return bubble_regions
            
        except Exception as e:
            self.logger.error(f"Error detecting bubble regions: {str(e)}")
            return []
    
    def _crop_bubble(self, image: np.ndarray, region: Dict[str, Any]) -> np.ndarray:
        """
        Crop bubble region from image with padding
        
        Args:
            image: Source image
            region: Bubble region coordinates
            
        Returns:
            Cropped bubble image
        """
        padding = self.omr_config['crop_padding']
        
        x1 = max(0, region['x'] - padding)
        y1 = max(0, region['y'] - padding)
        x2 = min(image.shape[1], region['x'] + region['width'] + padding)
        y2 = min(image.shape[0], region['y'] + region['height'] + padding)
        
        return image[y1:y2, x1:x2]
    
    def _classify_bubble(self, bubble_crop: np.ndarray) -> Dict[str, Any]:
        """
        Classify a single bubble crop using YOLO classification model
        
        Args:
            bubble_crop: Cropped bubble image
            
        Returns:
            Classification result
        """
        try:
            # Run YOLO classification
            results = self.model(bubble_crop, verbose=False)
            result = results[0]
            
            # Extract classification results
            if hasattr(result, 'probs') and result.probs is not None:
                # Classification model results
                class_id = result.probs.top1
                confidence = float(result.probs.top1conf)
                class_name = result.names[class_id] if hasattr(result, 'names') else self.class_names.get(class_id, 'unknown')
                
                # Get full probability distribution
                probs = result.probs.data.cpu().numpy() if hasattr(result.probs, 'data') else [confidence, 1-confidence]
                
                is_marked = class_name == 'marked_bubble'
                
                return {
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': confidence,
                    'is_marked': is_marked,
                    'probabilities': {
                        'marked_bubble': float(probs[0]) if class_name == 'marked_bubble' else float(probs[1]),
                        'unmarked_bubble': float(probs[1]) if class_name == 'marked_bubble' else float(probs[0])
                    },
                    'prediction_quality': 'high' if confidence > 0.8 else 'medium' if confidence > 0.6 else 'low'
                }
            else:
                # Fallback for unexpected result format
                return {
                    'class_id': 1,  # Default to unmarked
                    'class_name': 'unmarked_bubble',
                    'confidence': 0.5,
                    'is_marked': False,
                    'probabilities': {'marked_bubble': 0.0, 'unmarked_bubble': 1.0},
                    'prediction_quality': 'low',
                    'note': 'Fallback classification used'
                }
                
        except Exception as e:
            self.logger.error(f"Error classifying bubble: {str(e)}")
            return {
                'class_id': 1,
                'class_name': 'unmarked_bubble',
                'confidence': 0.0,
                'is_marked': False,
                'probabilities': {'marked_bubble': 0.0, 'unmarked_bubble': 0.0},
                'prediction_quality': 'error',
                'error': str(e)
            }
    
    def _organize_bubbles_by_questions(self, classified_bubbles: List[Dict[str, Any]], 
                                     image_shape: Tuple[int, int, int]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize classified bubbles into questions/rows using spatial clustering
        
        Args:
            classified_bubbles: List of classified bubble results
            image_shape: Shape of the original image
            
        Returns:
            Dictionary organized by question number
        """
        if not classified_bubbles:
            return {}
        
        try:
            # Extract positions for clustering
            positions = np.array([[b['position']['x'], b['position']['y']] for b in classified_bubbles])
            
            # Use DBSCAN clustering to group bubbles by rows
            clustering = DBSCAN(eps=self.omr_config['grid_tolerance'], min_samples=2)
            row_labels = clustering.fit_predict(positions)
            
            # Group bubbles by row
            organized = {}
            for i, bubble in enumerate(classified_bubbles):
                row_id = int(row_labels[i]) if row_labels[i] != -1 else f"isolated_{i}"
                
                if row_id not in organized:
                    organized[row_id] = []
                
                organized[row_id].append(bubble)
            
            # Sort bubbles within each row by x-coordinate (left to right)
            for row_id in organized:
                organized[row_id].sort(key=lambda x: x['position']['x'])
            
            # Convert to question format (Q1, Q2, etc.)
            question_organized = {}
            sorted_rows = sorted([k for k in organized.keys() if isinstance(k, int)])
            
            for i, row_id in enumerate(sorted_rows, 1):
                question_key = f"Q{i}"
                question_organized[question_key] = organized[row_id]
            
            return question_organized
            
        except Exception as e:
            self.logger.error(f"Error organizing bubbles by questions: {str(e)}")
            # Fallback: create single question with all bubbles
            return {"Q1": classified_bubbles}
    
    def _extract_student_answers_classification(self, organized_bubbles: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract student answers from organized classified bubbles
        
        Args:
            organized_bubbles: Bubbles organized by questions
            
        Returns:
            Student answers with confidence and validation info
        """
        student_answers = {}
        
        for question_id, bubbles in organized_bubbles.items():
            # Find marked bubbles in this question
            marked_bubbles = [b for b in bubbles if b['is_marked'] and b['confidence'] > self.omr_config['marking_threshold']]
            
            if len(marked_bubbles) == 0:
                # No answer marked
                student_answers[question_id] = {
                    'answer': None,
                    'confidence': 0.0,
                    'multiple_marks': False,
                    'bubble_count': len(bubbles),
                    'status': 'no_answer',
                    'raw_bubbles': bubbles
                }
            elif len(marked_bubbles) == 1:
                # Single answer (normal case)
                marked_bubble = marked_bubbles[0]
                # Determine answer choice (A, B, C, D) based on position
                answer_choice = self._determine_answer_choice(marked_bubble, bubbles)
                
                student_answers[question_id] = {
                    'answer': answer_choice,
                    'confidence': marked_bubble['confidence'],
                    'multiple_marks': False,
                    'bubble_count': len(bubbles),
                    'status': 'answered',
                    'raw_bubbles': bubbles
                }
            else:
                # Multiple marks detected
                highest_confidence_bubble = max(marked_bubbles, key=lambda x: x['confidence'])
                answer_choice = self._determine_answer_choice(highest_confidence_bubble, bubbles)
                
                student_answers[question_id] = {
                    'answer': answer_choice,
                    'confidence': highest_confidence_bubble['confidence'],
                    'multiple_marks': True,
                    'bubble_count': len(bubbles),
                    'status': 'multiple_marks',
                    'marked_count': len(marked_bubbles),
                    'raw_bubbles': bubbles
                }
        
        return student_answers
    
    def _determine_answer_choice(self, marked_bubble: Dict[str, Any], all_bubbles: List[Dict[str, Any]]) -> str:
        """
        Determine answer choice (A, B, C, D) based on bubble position
        
        Args:
            marked_bubble: The marked bubble
            all_bubbles: All bubbles in the question
            
        Returns:
            Answer choice letter
        """
        # Sort bubbles by x-coordinate (left to right)
        sorted_bubbles = sorted(all_bubbles, key=lambda x: x['position']['x'])
        
        # Find position of marked bubble
        for i, bubble in enumerate(sorted_bubbles):
            if bubble['bubble_id'] == marked_bubble['bubble_id']:
                # Convert to letter (0->A, 1->B, 2->C, 3->D, etc.)
                return chr(ord('A') + i)
        
        return 'A'  # Default fallback
    
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