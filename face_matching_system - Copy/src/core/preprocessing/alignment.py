import cv2
import numpy as np
import logging
import mediapipe as mp
from typing import Optional, Tuple, List, Dict, Any
import math

class AdvancedFaceAligner:
    """
    Advanced face alignment with GPU acceleration and multiple detection methods.
    Uses MediaPipe, OpenCV, and custom algorithms for robust face alignment.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (160, 160), 
                 use_gpu: bool = False, processing_config: Dict = None):
        self.target_size = target_size
        self.use_gpu = use_gpu
        self.processing_config = processing_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize detection methods
        self._init_mediapipe()
        self._init_opencv_detectors()
        
        # Face alignment parameters
        self.alignment_params = {
            'eye_distance_threshold': 0.1,
            'face_confidence_threshold': 0.7,
            'alignment_iterations': 3,
            'geometric_correction': True
        }
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh with optimized settings"""
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_face_detection = mp.solutions.face_detection
            
            # Face detection for quick face location
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full range model
                min_detection_confidence=0.5
            )
            
            # Face mesh for detailed landmarks
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.logger.info("✅ MediaPipe face detection and mesh initialized")
            self.mediapipe_available = True
            
        except Exception as e:
            self.logger.error(f"❌ MediaPipe initialization failed: {e}")
            self.mediapipe_available = False
    
    def _init_opencv_detectors(self):
        """Initialize OpenCV face detectors as fallback"""
        try:
            # Haar cascade detector
            self.haar_face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # DNN face detector (more accurate)
            try:
                self.dnn_net = cv2.dnn.readNetFromTensorflow(
                    'opencv_face_detector_uint8.pb',
                    'opencv_face_detector.pbtxt'
                )
                self.dnn_available = True
            except:
                self.dnn_available = False
                self.logger.info("DNN face detector not available, using Haar cascades")
            
            self.opencv_available = True
            self.logger.info("✅ OpenCV face detectors initialized")
            
        except Exception as e:
            self.logger.error(f"❌ OpenCV detector initialization failed: {e}")
            self.opencv_available = False
    
    def align_face_conservative(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Conservative face alignment that only corrects orientation and crops face region

        Args:
            image: Input image as numpy array

        Returns:
            Aligned image or None if alignment fails
        """
        # Try MediaPipe first (most accurate)
        if self.mediapipe_available:
            aligned = self._align_with_mediapipe_conservative(image)
            if aligned is not None:
                return aligned

        # Fallback to OpenCV DNN detector
        if self.dnn_available:
            aligned = self._align_with_dnn_conservative(image)
            if aligned is not None:
                return aligned

        # Final fallback to Haar cascades
        if self.opencv_available:
            aligned = self._align_with_haar_conservative(image)
            if aligned is not None:
                return aligned

        self.logger.warning("⚠️ All face alignment methods failed")
        return None

    def align_face_advanced(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Advanced face alignment using multiple methods with fallback
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Aligned image or None if alignment fails
        """
        # Try MediaPipe first (most accurate)
        if self.mediapipe_available:
            aligned = self._align_with_mediapipe(image)
            if aligned is not None:
                return self._post_process_alignment(aligned)
        
        # Fallback to OpenCV DNN detector
        if self.dnn_available:
            aligned = self._align_with_dnn(image)
            if aligned is not None:
                return self._post_process_alignment(aligned)
        
        # Final fallback to Haar cascades
        if self.opencv_available:
            aligned = self._align_with_haar(image)
            if aligned is not None:
                return self._post_process_alignment(aligned)
        
        self.logger.warning("⚠️ All face alignment methods failed")
        return None
    
    def _align_with_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Align face using MediaPipe landmarks"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Extract key facial landmarks
            key_points = self._extract_key_landmarks(landmarks, w, h)
            
            if key_points is None:
                return None
            
            # Perform geometric alignment
            aligned_image = self._perform_geometric_alignment(image, key_points)
            
            return aligned_image
            
        except Exception as e:
            self.logger.warning(f"⚠️ MediaPipe alignment failed: {e}")
            return None
    
    def _extract_key_landmarks(self, landmarks, w: int, h: int) -> Optional[Dict]:
        """Extract key facial landmarks for alignment"""
        try:
            # MediaPipe landmark indices for key facial features
            landmark_indices = {
                'left_eye_center': 33,
                'right_eye_center': 263,
                'nose_tip': 1,
                'left_mouth_corner': 61,
                'right_mouth_corner': 291,
                'left_eye_outer': 130,
                'right_eye_outer': 359,
                'chin_center': 175,
                'forehead_center': 9
            }
            
            key_points = {}
            for name, idx in landmark_indices.items():
                landmark = landmarks.landmark[idx]
                key_points[name] = [landmark.x * w, landmark.y * h]
            
            # Validate eye distance (quality check)
            eye_distance = np.linalg.norm(
                np.array(key_points['left_eye_center']) - np.array(key_points['right_eye_center'])
            )
            
            if eye_distance < w * self.alignment_params['eye_distance_threshold']:
                return None  # Face too small or detection error
            
            return key_points
            
        except Exception as e:
            self.logger.warning(f"Landmark extraction failed: {e}")
            return None
    
    def _perform_geometric_alignment(self, image: np.ndarray, key_points: Dict) -> np.ndarray:
        """Perform geometric alignment based on facial landmarks"""
        try:
            # Calculate face angle based on eye positions
            left_eye = np.array(key_points['left_eye_center'])
            right_eye = np.array(key_points['right_eye_center'])
            
            # Calculate rotation angle
            eye_vector = right_eye - left_eye
            angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
            
            # Calculate center point between eyes
            eye_center = (left_eye + right_eye) / 2
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
            
            # Apply rotation
            rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
            
            # Update landmark positions after rotation
            rotated_landmarks = self._transform_landmarks(key_points, rotation_matrix)
            
            # Calculate crop region for face extraction
            crop_region = self._calculate_face_crop_region(rotated_landmarks, rotated_image.shape)
            
            # Extract and resize face region
            cropped_face = rotated_image[
                crop_region['y']:crop_region['y'] + crop_region['h'],
                crop_region['x']:crop_region['x'] + crop_region['w']
            ]
            
            # Resize to target size
            aligned_face = cv2.resize(cropped_face, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            return aligned_face
            
        except Exception as e:
            self.logger.warning(f"Geometric alignment failed: {e}")
            return None
    
    def _transform_landmarks(self, landmarks: Dict, transform_matrix: np.ndarray) -> Dict:
        """Transform landmarks using transformation matrix"""
        transformed = {}
        for name, point in landmarks.items():
            # Convert to homogeneous coordinates
            point_homogeneous = np.array([point[0], point[1], 1])
            # Apply transformation
            transformed_point = transform_matrix @ point_homogeneous
            transformed[name] = [transformed_point[0], transformed_point[1]]
        return transformed
    
    def _calculate_face_crop_region(self, landmarks: Dict, image_shape: Tuple) -> Dict:
        """Calculate optimal crop region for face"""
        try:
            # Get face boundaries
            all_points = np.array(list(landmarks.values()))
            min_x, min_y = np.min(all_points, axis=0)
            max_x, max_y = np.max(all_points, axis=0)
            
            # Add margins (20% on each side)
            face_width = max_x - min_x
            face_height = max_y - min_y
            margin_x = face_width * 0.2
            margin_y = face_height * 0.2
            
            # Calculate crop region
            crop_x = max(0, int(min_x - margin_x))
            crop_y = max(0, int(min_y - margin_y))
            crop_w = min(image_shape[1] - crop_x, int(face_width + 2 * margin_x))
            crop_h = min(image_shape[0] - crop_y, int(face_height + 2 * margin_y))
            
            # Ensure square crop for consistent alignment
            crop_size = min(crop_w, crop_h)
            
            return {
                'x': crop_x,
                'y': crop_y,
                'w': crop_size,
                'h': crop_size
            }
            
        except Exception as e:
            self.logger.warning(f"Crop region calculation failed: {e}")
            return {'x': 0, 'y': 0, 'w': image_shape[1], 'h': image_shape[0]}
    
    def _align_with_dnn(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Align face using OpenCV DNN face detector"""
        try:
            h, w = image.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            
            # Find best detection
            best_confidence = 0
            best_detection = None
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > best_confidence and confidence > self.alignment_params['face_confidence_threshold']:
                    best_confidence = confidence
                    best_detection = detections[0, 0, i, 3:7]
            
            if best_detection is None:
                return None
            
            # Extract face region
            x1 = int(best_detection[0] * w)
            y1 = int(best_detection[1] * h)
            x2 = int(best_detection[2] * w)
            y2 = int(best_detection[3] * h)
            
            # Add margins and extract face
            margin = 0.2
            margin_x = int((x2 - x1) * margin)
            margin_y = int((y2 - y1) * margin)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)
            
            face_region = image[y1:y2, x1:x2]
            aligned_face = cv2.resize(face_region, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            return aligned_face
            
        except Exception as e:
            self.logger.warning(f"DNN alignment failed: {e}")
            return None
    
    def _align_with_haar(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Align face using Haar cascade detector"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            faces = self.haar_face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None
            
            # Use largest detected face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Add margins
            margin = 0.2
            margin_x = int(w * margin)
            margin_y = int(h * margin)
            
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(image.shape[1] - x, w + 2 * margin_x)
            h = min(image.shape[0] - y, h + 2 * margin_y)
            
            face_region = image[y:y+h, x:x+w]
            aligned_face = cv2.resize(face_region, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            return aligned_face
            
        except Exception as e:
            self.logger.warning(f"Haar alignment failed: {e}")
            return None
    
    def _align_with_mediapipe_conservative(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Conservative MediaPipe alignment - minimal rotation and cropping only"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
            results = self.face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]

            # Extract only eye landmarks for minimal rotation
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]

            left_eye_pos = [left_eye.x * w, left_eye.y * h]
            right_eye_pos = [right_eye.x * w, right_eye.y * h]

            # Calculate rotation angle (only correct if significantly tilted)
            eye_vector = np.array(right_eye_pos) - np.array(left_eye_pos)
            angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))

            # Only rotate if angle is significant (more than 5 degrees)
            if abs(angle) > 5:
                eye_center = (np.array(left_eye_pos) + np.array(right_eye_pos)) / 2
                rotation_matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
                rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
            else:
                rotated_image = image

            # Conservative face crop with minimal margins
            face_landmarks = []
            for landmark in landmarks.landmark:
                face_landmarks.append([landmark.x * w, landmark.y * h])

            face_landmarks = np.array(face_landmarks)
            min_x, min_y = np.min(face_landmarks, axis=0)
            max_x, max_y = np.max(face_landmarks, axis=0)

            # Small margin (10% instead of 20%)
            margin = 0.1
            face_width = max_x - min_x
            face_height = max_y - min_y
            margin_x = face_width * margin
            margin_y = face_height * margin

            crop_x = max(0, int(min_x - margin_x))
            crop_y = max(0, int(min_y - margin_y))
            crop_w = min(w - crop_x, int(face_width + 2 * margin_x))
            crop_h = min(h - crop_y, int(face_height + 2 * margin_y))

            face_region = rotated_image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            aligned_face = cv2.resize(face_region, self.target_size, interpolation=cv2.INTER_LANCZOS4)

            return aligned_face

        except Exception as e:
            self.logger.warning(f"⚠️ Conservative MediaPipe alignment failed: {e}")
            return None

    def _align_with_dnn_conservative(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Conservative DNN alignment - basic face detection and crop"""
        try:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()

            best_confidence = 0
            best_detection = None

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > best_confidence and confidence > 0.7:
                    best_confidence = confidence
                    best_detection = detections[0, 0, i, 3:7]

            if best_detection is None:
                return None

            x1 = int(best_detection[0] * w)
            y1 = int(best_detection[1] * h)
            x2 = int(best_detection[2] * w)
            y2 = int(best_detection[3] * h)

            # Minimal margins (10%)
            margin = 0.1
            margin_x = int((x2 - x1) * margin)
            margin_y = int((y2 - y1) * margin)

            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(w, x2 + margin_x)
            y2 = min(h, y2 + margin_y)

            face_region = image[y1:y2, x1:x2]
            aligned_face = cv2.resize(face_region, self.target_size, interpolation=cv2.INTER_LANCZOS4)

            return aligned_face

        except Exception as e:
            self.logger.warning(f"Conservative DNN alignment failed: {e}")
            return None

    def _align_with_haar_conservative(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Conservative Haar alignment - basic face detection and crop"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            faces = self.haar_face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                return None

            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face

            # Minimal margins (10%)
            margin = 0.1
            margin_x = int(w * margin)
            margin_y = int(h * margin)

            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(image.shape[1] - x, w + 2 * margin_x)
            h = min(image.shape[0] - y, h + 2 * margin_y)

            face_region = image[y:y+h, x:x+w]
            aligned_face = cv2.resize(face_region, self.target_size, interpolation=cv2.INTER_LANCZOS4)

            return aligned_face

        except Exception as e:
            self.logger.warning(f"Conservative Haar alignment failed: {e}")
            return None
    
    def _post_process_alignment(self, aligned_image: np.ndarray) -> np.ndarray:
        """Post-process aligned face for optimal quality"""
        try:
            # Apply slight Gaussian blur to reduce noise
            if self.processing_config.get('memory_optimization', False):
                # Light processing for memory-constrained systems
                processed = cv2.GaussianBlur(aligned_image, (3, 3), 0.5)
            else:
                # Full processing
                processed = cv2.GaussianBlur(aligned_image, (3, 3), 0.5)
                
                # Enhance contrast slightly
                lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return processed
            
        except Exception as e:
            self.logger.warning(f"Post-processing failed: {e}")
            return aligned_image
    
    def detect_orientation(self, image: np.ndarray) -> float:
        """Detect face orientation angle for correction"""
        try:
            if not self.mediapipe_available:
                return 0.0
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return 0.0
            
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Get eye positions
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]
            
            left_eye_pos = [left_eye.x * w, left_eye.y * h]
            right_eye_pos = [right_eye.x * w, right_eye.y * h]
            
            # Calculate angle
            eye_vector = np.array(right_eye_pos) - np.array(left_eye_pos)
            angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
            
            return angle
            
        except Exception as e:
            self.logger.warning(f"Orientation detection failed: {e}")
            return 0.0
    
    def get_alignment_capabilities(self) -> Dict[str, Any]:
        """Get current alignment capabilities"""
        return {
            'mediapipe_available': self.mediapipe_available,
            'opencv_dnn_available': self.dnn_available,
            'opencv_haar_available': self.opencv_available,
            'gpu_acceleration': self.use_gpu,
            'geometric_correction': self.alignment_params['geometric_correction'],
            'target_size': self.target_size,
            'processing_config': self.processing_config
        }