"""
Camera Service for OMR Flask Application
Handles camera capture from PC, phone, and AR devices
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
import platform


class CameraService:
    def __init__(self):
        """Initialize camera service"""
        self.camera = None
        self.camera_index = 0
        self.available_cameras = self._detect_cameras()
    
    def _detect_cameras(self) -> Dict[int, Dict[str, Any]]:
        """Detect available cameras on the system"""
        cameras = {}
        
        # Test up to 10 camera indices
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                cameras[i] = {
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'available': True
                }
                
                cap.release()
            else:
                cap.release()
        
        return cameras
    
    def get_camera_status(self) -> Dict[str, Any]:
        """Get current camera status and available devices"""
        return {
            'available_cameras': self.available_cameras,
            'current_camera': self.camera_index if self.camera else None,
            'camera_active': self.camera is not None and self.camera.isOpened(),
            'platform': platform.system(),
            'opencv_version': cv2.__version__
        }
    
    def initialize_camera(self, camera_index: int = 0, width: int = 1280, height: int = 720) -> bool:
        """Initialize camera with specified parameters"""
        try:
            if self.camera:
                self.camera.release()
            
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Enable auto-focus if available
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            self.camera_index = camera_index
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def capture_image(self, camera_type: str = 'pc') -> Optional[np.ndarray]:
        """
        Capture image from specified camera type
        
        Args:
            camera_type: Type of camera ('pc', 'phone', 'ar')
            
        Returns:
            Captured image as numpy array or None if failed
        """
        try:
            if camera_type == 'pc':
                return self._capture_from_pc_camera()
            elif camera_type == 'phone':
                return self._capture_from_phone_camera()
            elif camera_type == 'ar':
                return self._capture_from_ar_camera()
            else:
                raise ValueError(f"Unsupported camera type: {camera_type}")
                
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
    
    def _capture_from_pc_camera(self) -> Optional[np.ndarray]:
        """Capture image from PC webcam"""
        if not self.camera or not self.camera.isOpened():
            if not self.initialize_camera():
                return None
        
        # Capture multiple frames to ensure camera is ready
        for _ in range(5):
            ret, frame = self.camera.read()
        
        if ret:
            # Apply basic image enhancement
            enhanced_frame = self._enhance_image(frame)
            return enhanced_frame
        
        return None
    
    def _capture_from_phone_camera(self) -> Optional[np.ndarray]:
        """
        Capture image from phone camera (via web interface)
        This method assumes the image is provided via web API
        """
        # For phone cameras, we rely on the web interface to capture
        # and send the image data. This method would be called after
        # receiving image data from a mobile device.
        return self._capture_from_pc_camera()  # Fallback to PC camera
    
    def _capture_from_ar_camera(self) -> Optional[np.ndarray]:
        """
        Capture image from AR device camera
        This is a placeholder for AR camera integration
        """
        # AR camera integration would require specific AR SDK
        # For now, fallback to PC camera
        return self._capture_from_pc_camera()
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply basic image enhancement for better OMR processing"""
        try:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back and convert to BGR
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and sharpened (0.7 original, 0.3 sharpened)
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return result
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image
    
    def capture_video_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame for video preview"""
        if not self.camera or not self.camera.isOpened():
            if not self.initialize_camera():
                return None
        
        ret, frame = self.camera.read()
        if ret:
            return frame
        return None
    
    def start_video_stream(self):
        """Start video streaming for live preview"""
        if not self.camera or not self.camera.isOpened():
            if not self.initialize_camera():
                return False
        return True
    
    def stop_video_stream(self):
        """Stop video streaming"""
        if self.camera:
            self.camera.release()
            self.camera = None
    
    def get_camera_properties(self) -> Dict[str, Any]:
        """Get current camera properties"""
        if not self.camera or not self.camera.isOpened():
            return {}
        
        return {
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.camera.get(cv2.CAP_PROP_FPS)),
            'brightness': self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.camera.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.camera.get(cv2.CAP_PROP_SATURATION),
            'exposure': self.camera.get(cv2.CAP_PROP_EXPOSURE)
        }
    
    def set_camera_property(self, property_name: str, value: float) -> bool:
        """Set camera property"""
        if not self.camera or not self.camera.isOpened():
            return False
        
        property_map = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'saturation': cv2.CAP_PROP_SATURATION,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'focus': cv2.CAP_PROP_FOCUS,
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS
        }
        
        if property_name in property_map:
            return self.camera.set(property_map[property_name], value)
        
        return False
    
    def __del__(self):
        """Cleanup camera resources"""
        if self.camera:
            self.camera.release()