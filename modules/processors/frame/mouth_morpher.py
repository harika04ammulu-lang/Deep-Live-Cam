"""
Mouth Morphing Module for Real-Time Lip Sync Integration

Integrates advanced mouth morphing from lip_sync_advanced.py into the face swap pipeline.
Morphs the source face's mouth region to match the target face's mouth landmarks
BEFORE the INSwapper step, enabling more natural lip sync in real-time face swapping.

Key Features:
- Extracts mouth openness from target (webcam driver) face
- Applies EMA temporal smoothing to mouth openness
- Morphs source face mouth region using landmark-based warping
- Integrates seamlessly before INSwapper inference
- Preserves face identity while matching mouth movements
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from collections import deque
import threading

from modules.typing import Face, Frame
from modules.processors.frame.lip_sync_advanced import (
    calculate_mouth_shape_confidence,
    MOUTH_SHAPES
)

# Thread-safe lock for mouth morphing state
MOUTH_MORPHER_LOCK = threading.Lock()

# Global state for mouth openness smoothing
_mouth_openness_history = deque(maxlen=10)
_last_mouth_shape = "neutral"

# Mouth landmark indices (68-point landmarks)
MOUTH_LANDMARKS_68 = list(range(48, 68))  # Standard 68-point mouth indices


class MouthOpennessTracker:
    """Tracks and smooths mouth openness across frames for stable morphing."""
    
    def __init__(self, history_size: int = 10, smoothing_alpha: float = 0.8):
        """
        Initialize mouth openness tracker.
        
        Args:
            history_size: Number of frames to track
            smoothing_alpha: EMA smoothing factor (0.0-1.0, higher=smoother)
        """
        self.history = deque(maxlen=history_size)
        self.smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))
        self.current_openness = 0.0
        self.smoothed_openness = 0.0
        
    def update(self, openness: float) -> float:
        """
        Update with new openness value and return smoothed result.
        
        Args:
            openness: Raw mouth openness (0.0-1.0)
            
        Returns:
            Smoothed openness value
        """
        openness = max(0.0, min(1.0, openness))
        self.history.append(openness)
        self.current_openness = openness
        
        # Apply EMA smoothing
        if len(self.history) > 1:
            self.smoothed_openness = (
                self.smoothing_alpha * self.smoothed_openness +
                (1.0 - self.smoothing_alpha) * openness
            )
        else:
            self.smoothed_openness = openness
            
        return self.smoothed_openness
    
    def get_smoothed_openness(self) -> float:
        """Get current smoothed openness value."""
        return self.smoothed_openness
    
    def reset(self):
        """Reset tracker state."""
        self.history.clear()
        self.current_openness = 0.0
        self.smoothed_openness = 0.0


# Global tracker instance
_openness_tracker = MouthOpennessTracker()


def extract_mouth_landmarks_from_face(face: Face) -> Optional[np.ndarray]:
    """
    Extract mouth landmarks from a Face object.
    
    Supports both 5-point keypoints (kps) and 106-point landmarks.
    For 5-point, we estimate mouth position; for 106-point, we extract precise mouth region.
    
    Args:
        face: Face object with landmarks
        
    Returns:
        Mouth landmarks array (Nx2) or None if not available
    """
    # Try 106-point landmarks first (more accurate)
    if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
        landmarks = face.landmark_2d_106
        if landmarks.shape[0] >= 77:  # Has full mouth region (48-77)
            return landmarks[48:77]  # Return mouth region (outer + inner contours)
    
    # Fallback: estimate from 5-point keypoints
    if hasattr(face, 'kps') and face.kps is not None:
        kps = face.kps
        if kps.shape[0] >= 5:
            # 5-point: left_eye, right_eye, nose, left_mouth, right_mouth
            # Estimate mouth region from mouth corners
            left_mouth = kps[3]  # Left mouth corner
            right_mouth = kps[4]  # Right mouth corner
            
            # Create simple mouth representation (corners + center)
            mouth_center = (left_mouth + right_mouth) / 2.0
            mouth_landmarks = np.array([
                left_mouth,
                mouth_center,
                right_mouth
            ], dtype=np.float32)
            return mouth_landmarks
    
    return None


def calculate_mouth_openness_from_landmarks(mouth_landmarks: np.ndarray) -> float:
    """
    Calculate mouth openness from landmarks.
    
    Args:
        mouth_landmarks: Mouth landmark points (Nx2)
        
    Returns:
        Openness value (0.0=closed, 1.0=fully open)
    """
    if mouth_landmarks is None or mouth_landmarks.shape[0] < 3:
        return 0.0
    
    try:
        # For 3-point (simplified from 5-point keypoints)
        if mouth_landmarks.shape[0] == 3:
            # Estimate based on horizontal width only
            left_corner = mouth_landmarks[0]
            right_corner = mouth_landmarks[2]
            width = np.linalg.norm(right_corner - left_corner)
            # Normalized openness (assume typical mouth width is 50-80 pixels)
            openness = max(0.0, min(1.0, (width - 30) / 50.0))
            return openness
        
        # For full mouth landmarks (20+ points)
        if mouth_landmarks.shape[0] >= 20:
            # Calculate vertical distance (top lip to bottom lip)
            upper_lip_points = mouth_landmarks[2:7]  # Top lip
            lower_lip_points = mouth_landmarks[10:15]  # Bottom lip
            
            upper_center = np.mean(upper_lip_points, axis=0)
            lower_center = np.mean(lower_lip_points, axis=0)
            
            vertical_dist = np.linalg.norm(upper_center - lower_center)
            
            # Calculate horizontal width (mouth corners)
            left_corner = mouth_landmarks[0]
            right_corner = mouth_landmarks[6]
            horizontal_dist = np.linalg.norm(right_corner - left_corner)
            
            # Openness ratio
            if horizontal_dist > 0:
                openness = vertical_dist / horizontal_dist
            else:
                openness = 0.0
            
            # Normalize to 0-1 range (typical ratio is 0.0-0.8)
            openness = max(0.0, min(1.0, openness / 0.8))
            return openness
    except Exception as e:
        print(f"[MouthMorpher] Error calculating openness: {e}")
        return 0.0
    
    return 0.0


def morph_mouth_region_with_warping(
    aligned_face: np.ndarray,
    source_mouth_landmarks: np.ndarray,
    target_openness: float,
    strength: float = 0.7
) -> np.ndarray:
    """
    Morph the mouth region of an aligned face image using landmark-based warping.
    
    Args:
        aligned_face: Aligned face image (128x128x3 or similar)
        source_mouth_landmarks: Source mouth landmarks in aligned face coords
        target_openness: Target mouth openness (0.0-1.0)
        strength: Morphing strength (0.0-1.0)
        
    Returns:
        Morphed face image
    """
    if aligned_face is None or source_mouth_landmarks is None:
        return aligned_face
    
    if target_openness < 0.05:  # Mouth essentially closed, no morphing needed
        return aligned_face
    
    # SHAPE VALIDATION: Verify input is valid
    if not isinstance(aligned_face, np.ndarray):
        print("[MouthMorpher] Invalid aligned_face type - returning original")
        return aligned_face
    
    if aligned_face.size == 0:
        print("[MouthMorpher] Empty aligned_face - returning original")
        return aligned_face
    
    try:
        h, w = aligned_face.shape[:2]
        morphed_face = aligned_face.copy()
        
        # Create mouth region mask
        mouth_mask = np.zeros((h, w), dtype=np.uint8)
        
        if source_mouth_landmarks.shape[0] >= 3:
            # Convert landmarks to image coordinates
            pts = source_mouth_landmarks.astype(np.int32)
            pts = np.clip(pts, 0, [w-1, h-1])
            
            # Draw filled polygon for mouth region
            if pts.shape[0] >= 3:
                cv2.fillPoly(mouth_mask, [pts], 255)
                
                # Expand mask slightly for smoother transition
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=1)
                
                # Feather the mask edges
                mouth_mask = cv2.GaussianBlur(mouth_mask, (15, 15), 5)
                
                # Calculate vertical deformation based on openness
                # Create deformation field for vertical expansion
                center_y = int(np.mean(pts[:, 1]))
                
                # Apply vertical scaling to open mouth
                scale_factor = 1.0 + (target_openness * strength * 0.3)  # Up to 30% expansion
                
                # Create meshgrid for warping
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                
                # Apply deformation in mouth region
                # Push lower lip down, upper lip stays relatively stable
                for y in range(h):
                    for x in range(w):
                        if mouth_mask[y, x] > 10:
                            mask_weight = mouth_mask[y, x] / 255.0
                            
                            # Vertical displacement based on distance from center
                            dist_from_center = (y - center_y) / (h / 4.0)
                            
                            # Lower part moves down, upper part moves up slightly
                            if dist_from_center > 0:  # Lower lip
                                displacement = int(dist_from_center * target_openness * strength * 8)
                            else:  # Upper lip
                                displacement = int(dist_from_center * target_openness * strength * 3)
                            
                            # Sample from displaced position
                            src_y = max(0, min(h-1, y - displacement))
                            
                            # SHAPE VALIDATION: Ensure safe array access
                            if src_y >= 0 and src_y < h and x >= 0 and x < w:
                                try:
                                    morphed_face[y, x] = (
                                        aligned_face[src_y, x] * mask_weight +
                                        aligned_face[y, x] * (1 - mask_weight)
                                    )
                                except (ValueError, IndexError):
                                    # Shape mismatch or index error - skip this pixel
                                    pass
        
        return morphed_face
        
    except Exception as e:
        print(f"[MouthMorpher] Error during warping: {e}")
        return aligned_face


def apply_mouth_morphing_to_source(
    source_face: Face,
    target_face: Face,
    temp_frame: Frame,
    strength: float = 0.7,
    smoothing_alpha: float = 0.8
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply mouth morphing to the source face to match target face's mouth movement.
    
    This function:
    1. Extracts mouth landmarks from target (webcam driver) face
    2. Calculates mouth openness with EMA smoothing
    3. Creates an aligned source face crop with morphed mouth
    4. Returns the morphed crop ready for INSwapper
    
    Args:
        source_face: Static source face
        target_face: Real-time target face (webcam driver)
        temp_frame: Current frame
        strength: Morphing strength (0.0-1.0)
        smoothing_alpha: EMA smoothing factor (0.0-1.0)
        
    Returns:
        Tuple of (morphed_aligned_source, affine_matrix) or (None, None)
    """
    global _openness_tracker
    
    if source_face is None or target_face is None:
        return None, None
    
    try:
        # Extract target mouth landmarks
        target_mouth_landmarks = extract_mouth_landmarks_from_face(target_face)
        
        if target_mouth_landmarks is None:
            return None, None
        
        # Calculate target mouth openness
        target_openness = calculate_mouth_openness_from_landmarks(target_mouth_landmarks)
        
        # Apply temporal smoothing
        with MOUTH_MORPHER_LOCK:
            smoothed_openness = _openness_tracker.update(target_openness)
        
        # If mouth is barely open, skip morphing
        if smoothed_openness < 0.05:
            return None, None
        
        # Get aligned source face (simulate what INSwapper would receive)
        # This is the 128x128 aligned crop before model inference
        from insightface.utils import face_align
        
        # Get target keypoints for alignment (to match INSwapper's alignment)
        if not hasattr(target_face, 'kps') or target_face.kps is None:
            return None, None
        
        target_kps = target_face.kps
        
        # Align source face using target's alignment (this gives us source appearance in target pose)
        # Standard alignment for 128x128 crop
        arcface_dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # Get affine matrix from source keypoints to standard template
        if not hasattr(source_face, 'kps') or source_face.kps is None:
            return None, None
        
        source_kps = source_face.kps
        
        # For morphing, we need source face aligned to a standard pose
        # Use the source's own keypoints for alignment
        M_source = cv2.estimateAffinePartial2D(
            source_kps,
            arcface_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )[0]
        
        if M_source is None:
            return None, None
        
        # Create aligned source face crop (128x128)
        aligned_source = cv2.warpAffine(
            temp_frame,
            M_source,
            (128, 128),
            borderValue=0.0
        )
        
        # Extract source mouth landmarks in aligned space
        source_mouth_landmarks_aligned = None
        if hasattr(source_face, 'landmark_2d_106') and source_face.landmark_2d_106 is not None:
            source_landmarks = source_face.landmark_2d_106[48:77]  # Mouth region
            # Transform to aligned space
            ones = np.ones((source_landmarks.shape[0], 1))
            source_landmarks_homo = np.hstack([source_landmarks, ones])
            source_mouth_landmarks_aligned = (M_source @ source_landmarks_homo.T).T
        
        # Apply mouth morphing to the aligned source face
        if source_mouth_landmarks_aligned is not None:
            morphed_aligned = morph_mouth_region_with_warping(
                aligned_source,
                source_mouth_landmarks_aligned,
                smoothed_openness,
                strength=strength
            )
            
            return morphed_aligned, M_source
        else:
            # Fallback: no morphing if landmarks unavailable
            return aligned_source, M_source
            
    except Exception as e:
        print(f"[MouthMorpher] Error in apply_mouth_morphing_to_source: {e}")
        return None, None


def reset_mouth_morpher():
    """Reset the mouth morpher state (call when switching faces or videos)."""
    global _openness_tracker
    with MOUTH_MORPHER_LOCK:
        _openness_tracker.reset()
        _mouth_openness_history.clear()


def get_mouth_openness_smoothed() -> float:
    """Get the current smoothed mouth openness value (for debugging/monitoring)."""
    global _openness_tracker
    with MOUTH_MORPHER_LOCK:
        return _openness_tracker.get_smoothed_openness()
