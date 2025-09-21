"""
Preprocessing Service for OMR Flask Application
Handles image preprocessing for optimal YOLO model performance
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any, List
from skimage import morphology, filters
from skimage.restoration import denoise_bilateral
import logging


class PreprocessingService:
    def __init__(self):
        """Initialize preprocessing service"""
        self.logger = logging.getLogger(__name__)
    
    def preprocess_omr_image(self, image: np.ndarray, config: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Comprehensive preprocessing pipeline for OMR images
        
        Args:
            image: Input image
            config: Preprocessing configuration
            
        Returns:
            Tuple of (processed_image, preprocessing_info)
        """
        if config is None:
            config = self._get_default_config()
        
        preprocessing_info = {
            'original_shape': image.shape,
            'steps_applied': [],
            'parameters_used': config.copy(),
            'quality_metrics': {}
        }
        
        try:
            processed_image = image.copy()
            
            # Step 1: Image validation and basic corrections
            processed_image, validation_info = self._validate_and_correct_image(processed_image)
            preprocessing_info['steps_applied'].append('validation_and_correction')
            preprocessing_info['validation_info'] = validation_info
            
            # Step 2: Noise reduction
            if config.get('noise_reduction', True):
                processed_image = self._apply_noise_reduction(processed_image)
                preprocessing_info['steps_applied'].append('noise_reduction')
            
            # Step 3: Lighting normalization
            if config.get('lighting_normalization', True):
                processed_image = self._normalize_lighting(processed_image)
                preprocessing_info['steps_applied'].append('lighting_normalization')
            
            # Step 4: Contrast enhancement
            if config.get('contrast_enhancement', True):
                processed_image = self._enhance_contrast(processed_image)
                preprocessing_info['steps_applied'].append('contrast_enhancement')
            
            # Step 5: Geometric corrections
            if config.get('geometric_correction', True):
                processed_image, correction_info = self._apply_geometric_corrections(processed_image)
                preprocessing_info['steps_applied'].append('geometric_correction')
                preprocessing_info['geometric_correction_info'] = correction_info
            
            # Step 6: Document enhancement for OMR
            if config.get('omr_enhancement', True):
                processed_image = self._enhance_for_omr(processed_image)
                preprocessing_info['steps_applied'].append('omr_enhancement')
            
            # Step 7: Final quality checks
            quality_metrics = self._calculate_quality_metrics(processed_image, image)
            preprocessing_info['quality_metrics'] = quality_metrics
            
            # Step 8: Resize if needed
            if config.get('resize_target'):
                target_size = config['resize_target']
                processed_image = cv2.resize(processed_image, target_size, interpolation=cv2.INTER_LANCZOS4)
                preprocessing_info['steps_applied'].append('resize')
                preprocessing_info['final_shape'] = processed_image.shape
            
            return processed_image, preprocessing_info
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            preprocessing_info['error'] = str(e)
            return image, preprocessing_info
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default preprocessing configuration"""
        return {
            'noise_reduction': True,
            'lighting_normalization': True,
            'contrast_enhancement': True,
            'geometric_correction': True,
            'omr_enhancement': True,
            'resize_target': None,
            'blur_kernel_size': 3,
            'morphology_kernel_size': 3,
            'clahe_clip_limit': 3.0,
            'clahe_tile_size': (8, 8),
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75
        }
    
    def _validate_and_correct_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Validate image and apply basic corrections"""
        validation_info = {
            'original_channels': len(image.shape),
            'corrections_applied': []
        }
        
        # Ensure image is in color format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            validation_info['corrections_applied'].append('converted_to_color')
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            validation_info['corrections_applied'].append('removed_alpha_channel')
        
        # Check for extreme brightness/darkness
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        validation_info['mean_brightness'] = float(mean_brightness)
        
        if mean_brightness < 50:  # Very dark image
            image = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
            validation_info['corrections_applied'].append('brightness_correction_dark')
        elif mean_brightness > 200:  # Very bright image
            image = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
            validation_info['corrections_applied'].append('brightness_correction_bright')
        
        return image, validation_info
    
    def _apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction techniques"""
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Apply median filter for salt-and-pepper noise
        denoised = cv2.medianBlur(denoised, 3)
        
        return denoised
    
    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """Normalize lighting conditions across the image"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge channels back
        lab_clahe = cv2.merge([l_clahe, a, b])
        normalized = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return normalized
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better feature detection"""
        # Convert to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Apply histogram equalization to Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # Apply adaptive sharpening
        kernel = np.array([[-1, -1, -1, -1, -1],
                          [-1,  2,  2,  2, -1],
                          [-1,  2,  8,  2, -1],
                          [-1,  2,  2,  2, -1],
                          [-1, -1, -1, -1, -1]]) / 8.0
        
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Blend original and sharpened
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def _apply_geometric_corrections(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply geometric corrections like rotation and perspective correction"""
        correction_info = {
            'rotation_angle': 0,
            'perspective_corrected': False,
            'corrections_applied': []
        }
        
        try:
            # Detect and correct rotation
            corrected_image, rotation_angle = self._correct_rotation(image)
            if abs(rotation_angle) > 0.5:  # Only apply if significant rotation
                image = corrected_image
                correction_info['rotation_angle'] = float(rotation_angle)
                correction_info['corrections_applied'].append('rotation_correction')
            
            # Detect and correct perspective distortion
            perspective_corrected, perspective_applied = self._correct_perspective(image)
            if perspective_applied:
                image = perspective_corrected
                correction_info['perspective_corrected'] = True
                correction_info['corrections_applied'].append('perspective_correction')
            
        except Exception as e:
            self.logger.warning(f"Geometric correction failed: {str(e)}")
            correction_info['error'] = str(e)
        
        return image, correction_info
    
    def _correct_rotation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct document rotation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                # Focus on angles close to horizontal
                if abs(angle) < 45:
                    angles.append(angle)
            
            if angles:
                # Use median angle to avoid outliers
                rotation_angle = np.median(angles)
                
                # Rotate image
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                
                # Calculate new image dimensions
                cos_angle = abs(rotation_matrix[0, 0])
                sin_angle = abs(rotation_matrix[0, 1])
                new_width = int(height * sin_angle + width * cos_angle)
                new_height = int(height * cos_angle + width * sin_angle)
                
                # Adjust rotation matrix for new center
                rotation_matrix[0, 2] += (new_width - width) / 2
                rotation_matrix[1, 2] += (new_height - height) / 2
                
                rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                
                return rotated, rotation_angle
        
        return image, 0.0
    
    def _correct_perspective(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Detect and correct perspective distortion"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangular contour (likely the document)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If we found a quadrilateral that's large enough
            if len(approx) == 4 and cv2.contourArea(approx) > image.shape[0] * image.shape[1] * 0.3:
                # Order points: top-left, top-right, bottom-right, bottom-left
                points = self._order_points(approx.reshape(4, 2))
                
                # Calculate destination points for perspective correction
                width = max(
                    np.linalg.norm(points[1] - points[0]),
                    np.linalg.norm(points[2] - points[3])
                )
                height = max(
                    np.linalg.norm(points[3] - points[0]),
                    np.linalg.norm(points[2] - points[1])
                )
                
                dst_points = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype=np.float32)
                
                # Apply perspective transformation
                perspective_matrix = cv2.getPerspectiveTransform(points.astype(np.float32), dst_points)
                corrected = cv2.warpPerspective(image, perspective_matrix, (int(width), int(height)))
                
                return corrected, True
        
        return image, False
    
    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """Order points in clockwise order starting from top-left"""
        # Calculate sum and difference of coordinates
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        # Top-left: smallest sum, Top-right: smallest difference
        # Bottom-right: largest sum, Bottom-left: largest difference
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = points[np.argmin(s)]      # Top-left
        ordered[1] = points[np.argmin(diff)]   # Top-right
        ordered[2] = points[np.argmax(s)]      # Bottom-right
        ordered[3] = points[np.argmax(diff)]   # Bottom-left
        
        return ordered
    
    def _enhance_for_omr(self, image: np.ndarray) -> np.ndarray:
        """Apply OMR-specific enhancements"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to highlight bubbles
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Convert back to color for consistency
        enhanced = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        # Blend with original image to preserve some color information
        blended = cv2.addWeighted(image, 0.3, enhanced, 0.7, 0)
        
        return blended
    
    def _calculate_quality_metrics(self, processed_image: np.ndarray, original_image: np.ndarray) -> Dict[str, float]:
        """Calculate image quality metrics"""
        processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(processed_gray, cv2.CV_64F).var()
        
        # Calculate contrast (standard deviation)
        contrast = processed_gray.std()
        
        # Calculate brightness (mean intensity)
        brightness = processed_gray.mean()
        
        # Calculate structural similarity
        from skimage.metrics import structural_similarity as ssim
        similarity = ssim(original_gray, processed_gray)
        
        return {
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'brightness': float(brightness),
            'similarity_to_original': float(similarity)
        }
    
    def create_preprocessing_preview(self, image: np.ndarray, config: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """Create preview of preprocessing steps for visualization"""
        if config is None:
            config = self._get_default_config()
        
        previews = {'original': image.copy()}
        current_image = image.copy()
        
        # Step by step preview
        if config.get('noise_reduction', True):
            current_image = self._apply_noise_reduction(current_image)
            previews['noise_reduced'] = current_image.copy()
        
        if config.get('lighting_normalization', True):
            current_image = self._normalize_lighting(current_image)
            previews['lighting_normalized'] = current_image.copy()
        
        if config.get('contrast_enhancement', True):
            current_image = self._enhance_contrast(current_image)
            previews['contrast_enhanced'] = current_image.copy()
        
        if config.get('omr_enhancement', True):
            current_image = self._enhance_for_omr(current_image)
            previews['omr_enhanced'] = current_image.copy()
        
        previews['final'] = current_image
        
        return previews