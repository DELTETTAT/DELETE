import cv2
import numpy as np
import logging
import mediapipe as mp
from typing import Optional, Tuple

class FaceAligner:
    """
    Face alignment module using MediaPipe landmarks.
    Aligns faces to standardized positions for better recognition accuracy.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (160, 160)):
        self.target_size = target_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe Face Mesh
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.logger.info("✅ MediaPipe face mesh initialized")
        except Exception as e:
            self.logger.error(f"❌ MediaPipe initialization failed: {e}")
            self.face_mesh = None
    
    def align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Align face using MediaPipe landmarks
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Aligned image or None if alignment fails
        """
        if self.face_mesh is None:
            return None
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
            results = self.face_mesh.process(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                return None
            
            # Get landmarks for the first detected face
            landmarks = results.multi_face_landmarks[0]
            
            # Key points for alignment (left eye, right eye, nose tip, mouth corners)
            key_points = [
                33,   # Left eye corner
                263,  # Right eye corner
                1,    # Nose tip
                61,   # Left mouth corner
                291   # Right mouth corner
            ]
            
            # Extract coordinates
            h, w = image.shape[:2]
            points = []
            for idx in key_points:
                landmark = landmarks.landmark[idx]
                points.append([landmark.x * w, landmark.y * h])
            
            points = np.array(points, dtype=np.float32)
            
            # Define ideal points for alignment
            ideal_points = np.array([
                [0.35, 0.35],  # Left eye
                [0.65, 0.35],  # Right eye
                [0.5, 0.5],    # Nose
                [0.35, 0.75],  # Left mouth
                [0.65, 0.75]   # Right mouth
            ], dtype=np.float32) * min(self.target_size)
            
            # Calculate transformation matrix
            transform_matrix = cv2.estimateAffinePartial2D(points, ideal_points)[0]
            
            if transform_matrix is not None:
                # Apply transformation
                aligned_image = cv2.warpAffine(
                    image, transform_matrix, self.target_size,
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )
                return aligned_image
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Face alignment failed: {e}")
            return None
    
    def get_face_landmarks(self, image: np.ndarray) -> Optional[dict]:
        """Extract face landmarks for analysis"""
        if self.face_mesh is None:
            return None
        
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
            results = self.face_mesh.process(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # Extract key landmarks
            key_landmarks = {
                'left_eye': [landmarks.landmark[33].x * w, landmarks.landmark[33].y * h],
                'right_eye': [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h],
                'nose_tip': [landmarks.landmark[1].x * w, landmarks.landmark[1].y * h],
                'left_mouth': [landmarks.landmark[61].x * w, landmarks.landmark[61].y * h],
                'right_mouth': [landmarks.landmark[291].x * w, landmarks.landmark[291].y * h]
            }
            
            return key_landmarks
            
        except Exception as e:
            self.logger.error(f"Failed to extract landmarks: {e}")
            return None
