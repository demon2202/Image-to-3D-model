"""
Feature detection module supporting SIFT and ORB detectors.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class FeatureData:
    """Container for detected features."""
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    image_shape: Tuple[int, int]
    
    @property
    def num_features(self) -> int:
        return len(self.keypoints)
    
    def get_points(self) -> np.ndarray:
        """Return keypoint coordinates as Nx2 array."""
        return np.array([kp.pt for kp in self.keypoints], dtype=np.float64)


class FeatureDetector:
    """
    Multi-algorithm feature detector with configurable parameters.
    """
    
    def __init__(self, method: str = "sift", max_keypoints: int = 8000):
        self.method = method.lower()
        self.max_keypoints = max_keypoints
        self._detector = self._create_detector()
    
    def _create_detector(self):
        if self.method == "sift":
            return cv2.SIFT_create(nfeatures=self.max_keypoints)
        elif self.method == "orb":
            return cv2.ORB_create(nfeatures=self.max_keypoints)
        else:
            raise ValueError(f"Unknown detector: {self.method}. Use 'sift' or 'orb'.")
    
    def detect(self, image: np.ndarray) -> FeatureData:
        """
        Detect features in an image.
        
        Args:
            image: BGR or grayscale image (numpy array)
            
        Returns:
            FeatureData containing keypoints and descriptors
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        
        if descriptors is None:
            return FeatureData(
                keypoints=[],
                descriptors=np.array([]),
                image_shape=gray.shape[:2]
            )
        
        # Sort by response and keep top N
        if len(keypoints) > self.max_keypoints:
            indices = np.argsort([-kp.response for kp in keypoints])[:self.max_keypoints]
            keypoints = [keypoints[i] for i in indices]
            descriptors = descriptors[indices]
        
        return FeatureData(
            keypoints=keypoints,
            descriptors=descriptors,
            image_shape=gray.shape[:2]
        )
    
    def detect_batch(self, images: List[np.ndarray]) -> List[FeatureData]:
        """Detect features in multiple images."""
        return [self.detect(img) for img in images]