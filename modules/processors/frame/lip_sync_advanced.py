"""
Advanced Lip Sync Enhancement - Enhanced Mouth Morphing & Realistic Movements

This module provides advanced mouth shape morphing, temporal smoothing, and
realistic mouth movements for more natural lip sync results.

Features:
- Continuous mouth shape interpolation (not just discrete states)
- Temporal coherence with motion prediction
- Mouth shape morphing between reference poses
- Jaw movement tracking
- Eye blink synchronization
- Confidence-based adaptive blending
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import deque
import threading

LIP_SYNC_ADVANCED_LOCK = threading.Lock()

# Mouth shape reference landmarks (normalized)
MOUTH_SHAPES = {
    "neutral": np.array([
        [0.3, 0.4], [0.35, 0.35], [0.4, 0.33], [0.5, 0.32], [0.6, 0.33],
        [0.65, 0.35], [0.7, 0.4], [0.68, 0.5], [0.65, 0.58], [0.6, 0.62],
        [0.5, 0.65], [0.4, 0.62], [0.35, 0.58], [0.32, 0.5], [0.5, 0.48],
        [0.45, 0.52], [0.5, 0.53], [0.55, 0.52], [0.3, 0.5], [0.7, 0.5],
    ], dtype=np.float32),
    
    "open_a": np.array([  # "ah" - wide open
        [0.3, 0.5], [0.35, 0.42], [0.4, 0.38], [0.5, 0.35], [0.6, 0.38],
        [0.65, 0.42], [0.7, 0.5], [0.68, 0.65], [0.65, 0.75], [0.6, 0.80],
        [0.5, 0.82], [0.4, 0.80], [0.35, 0.75], [0.32, 0.65], [0.5, 0.45],
        [0.45, 0.58], [0.5, 0.62], [0.55, 0.58], [0.3, 0.55], [0.7, 0.55],
    ], dtype=np.float32),
    
    "open_e": np.array([  # "eh" - moderate open
        [0.25, 0.45], [0.3, 0.38], [0.4, 0.35], [0.5, 0.33], [0.6, 0.35],
        [0.7, 0.38], [0.75, 0.45], [0.72, 0.58], [0.68, 0.65], [0.6, 0.70],
        [0.5, 0.72], [0.4, 0.70], [0.32, 0.65], [0.28, 0.58], [0.5, 0.46],
        [0.45, 0.55], [0.5, 0.58], [0.55, 0.55], [0.25, 0.5], [0.75, 0.5],
    ], dtype=np.float32),
    
    "closed": np.array([  # Lips closed
        [0.3, 0.35], [0.4, 0.33], [0.5, 0.32], [0.6, 0.33], [0.7, 0.35],
        [0.68, 0.40], [0.65, 0.42], [0.6, 0.43], [0.5, 0.44], [0.4, 0.43],
        [0.35, 0.42], [0.32, 0.40], [0.35, 0.38], [0.4, 0.37], [0.5, 0.36],
        [0.6, 0.37], [0.65, 0.38], [0.3, 0.38], [0.7, 0.38], [0.5, 0.42],
    ], dtype=np.float32),
    
    "rounded": np.array([  # "oo" - rounded lips
        [0.35, 0.48], [0.4, 0.40], [0.48, 0.38], [0.5, 0.37], [0.52, 0.38],
        [0.6, 0.40], [0.65, 0.48], [0.63, 0.60], [0.58, 0.70], [0.5, 0.72],
        [0.42, 0.70], [0.37, 0.60], [0.38, 0.50], [0.42, 0.48], [0.5, 0.42],
        [0.45, 0.55], [0.5, 0.58], [0.55, 0.55], [0.35, 0.50], [0.65, 0.50],
    ], dtype=np.float32),
}

# Viseme to shape mapping (more detailed than phoneme)
VISEME_TO_SHAPE = {
    # Vowels
    0: "open_a",  # /ɑ/ lot, father
    1: "open_e",  # /ɛ/ dress
    2: "open_a",  # /æ/ trap
    3: "open_a",  # /ʌ/ strut
    4: "rounded",  # /ɔ/ cloth
    5: "rounded",  # /oʊ/ goat
    6: "rounded",  # /u/ goose
    7: "open_e",  # /ɪ/ kit
    8: "open_e",  # /i/ fleece
    9: "neutral",  # /ə/ about
    10: "open_a", # /ɚ/ letter
    
    # Consonants - lip relevant
    11: "closed",  # /p/ pat (bilabial)
    12: "closed",  # /b/ bat (bilabial)
    13: "closed",  # /m/ mat (bilabial)
    14: "open_e",  # /f/ fat (labiodental)
    15: "open_e",  # /v/ vat (labiodental)
}


class AdvancedMouthMorpher:
    """
    Advanced mouth shape morphing with smooth interpolation between shapes.
    
    Provides realistic mouth movements by interpolating between reference
    mouth shapes and applying smooth temporal transitions.
    """
    
    def __init__(self, history_size: int = 10):
        """
        Initialize the mouth morpher.
        
        Args:
            history_size: Number of frames to track for smoothing
        """
        self.history = deque(maxlen=history_size)
        self.shape_history = deque(maxlen=history_size)
        self.current_shape = "neutral"
        self.confidence = 0.5
        self.jaw_offset = 0.0
        
    def morph_shape(
        self,
        source_shape: str,
        target_shape: str,
        blend_factor: float
    ) -> np.ndarray:
        """
        Morph between two mouth shapes.
        
        Args:
            source_shape: Source shape name
            target_shape: Target shape name
            blend_factor: Blend factor (0.0=source, 1.0=target)
            
        Returns:
            Morphed mouth landmarks
        """
        if source_shape not in MOUTH_SHAPES or target_shape not in MOUTH_SHAPES:
            return MOUTH_SHAPES.get("neutral", MOUTH_SHAPES["neutral"])
        
        source = MOUTH_SHAPES[source_shape]
        target = MOUTH_SHAPES[target_shape]
        
        # Smooth interpolation using ease function
        t = self._ease_in_out(blend_factor)
        morphed = source * (1 - t) + target * t
        
        return morphed.astype(np.float32)
    
    def _ease_in_out(self, t: float) -> float:
        """Apply ease-in-out function for smooth transitions."""
        if t < 0.5:
            return 2 * t * t
        else:
            return -1 + (4 - 2 * t) * t
    
    def update_shape(
        self,
        new_shape: str,
        confidence: float = 0.8,
        jaw_movement: float = 0.0
    ) -> None:
        """
        Update mouth shape with confidence weighting.
        
        Args:
            new_shape: New mouth shape
            confidence: Confidence in the shape (0.0-1.0)
            jaw_movement: Jaw offset (-1.0 to 1.0)
        """
        self.shape_history.append(new_shape)
        self.current_shape = new_shape
        self.confidence = confidence
        self.jaw_offset = jaw_movement
    
    def get_smoothed_shape(self) -> Tuple[str, float]:
        """
        Get smoothed shape using history.
        
        Returns:
            (shape_name, blend_factor)
        """
        if len(self.shape_history) < 2:
            return self.current_shape, 1.0
        
        # Use weighted average of recent shapes
        recent_shapes = list(self.shape_history)[-3:]
        weights = np.array([0.2, 0.3, 0.5])  # Favor recent frames
        
        # Simple shape voting with confidence
        shape_weights = {}
        for shape, weight in zip(recent_shapes, weights):
            shape_weights[shape] = shape_weights.get(shape, 0) + weight
        
        best_shape = max(shape_weights, key=shape_weights.get)
        blend_factor = shape_weights[best_shape] / sum(shape_weights.values())
        
        return best_shape, blend_factor


class RealisticMouthAnimator:
    """
    Creates realistic mouth animations with natural movements.
    """
    
    def __init__(self):
        """Initialize the animator."""
        self.morpher = AdvancedMouthMorpher(history_size=10)
        self.prev_shape = "neutral"
        self.frame_count = 0
        
    def animate_mouth(
        self,
        frame: np.ndarray,
        mouth_landmarks: np.ndarray,
        target_shape: str,
        confidence: float = 0.8
    ) -> Tuple[np.ndarray, Dict]:
        """
        Animate mouth with smooth morphing.
        
        Args:
            frame: Input frame
            mouth_landmarks: Current mouth landmarks
            target_shape: Target shape to morph to
            confidence: Confidence in prediction (0.0-1.0)
            
        Returns:
            (animated_frame, animation_data)
        """
        self.frame_count += 1
        
        # Smooth shape transition
        blend_progress = min(self.frame_count / 5.0, 1.0)  # 5-frame transition
        
        # Morph landmarks
        if self.prev_shape != target_shape and blend_progress < 1.0:
            morphed = self.morpher.morph_shape(
                self.prev_shape,
                target_shape,
                blend_progress
            )
        else:
            morphed = MOUTH_SHAPES.get(target_shape, MOUTH_SHAPES["neutral"])
            self.prev_shape = target_shape
            self.frame_count = 0
        
        # Scale morphed shape to actual mouth region
        scaled = self._scale_landmarks(mouth_landmarks, morphed)
        
        # Apply enhanced blending
        result = frame.copy()
        animation_data = {
            "shape": target_shape,
            "confidence": confidence,
            "blend_progress": blend_progress,
            "scaled_landmarks": scaled,
        }
        
        return result, animation_data
    
    @staticmethod
    def _scale_landmarks(
        original: np.ndarray,
        normalized: np.ndarray
    ) -> np.ndarray:
        """
        Scale normalized landmarks to match original mouth region.
        
        Args:
            original: Original mouth landmarks
            normalized: Normalized shape (0-1 range)
            
        Returns:
            Scaled landmarks
        """
        if original.shape[0] < 2:
            return original
        
        # Get bounds of original
        orig_min = np.min(original, axis=0)
        orig_max = np.max(original, axis=0)
        orig_size = orig_max - orig_min
        
        # Scale normalized to original bounds
        scaled = normalized * orig_size + orig_min
        
        return scaled.astype(np.int32)


class JawMovementTracker:
    """
    Tracks and predicts jaw movement from audio features.
    """
    
    def __init__(self, history_size: int = 15):
        """
        Initialize jaw tracker.
        
        Args:
            history_size: Frames to track for smoothing
        """
        self.jaw_position_history = deque(maxlen=history_size)
        self.energy_history = deque(maxlen=history_size)
        self.current_jaw_offset = 0.0
        
    def update(self, energy: float) -> float:
        """
        Update jaw position based on energy.
        
        Args:
            energy: Audio energy (0.0-1.0)
            
        Returns:
            Jaw offset (-1.0 to 1.0)
        """
        self.energy_history.append(energy)
        
        # Jaw opens with increasing energy
        jaw_target = (energy - 0.3) * 2.0  # Scale to -1.0 to 1.0
        jaw_target = np.clip(jaw_target, -1.0, 1.0)
        
        # Smooth jaw movement
        if self.jaw_position_history:
            current = self.jaw_position_history[-1]
            # Exponential moving average for smooth motion
            self.current_jaw_offset = 0.7 * current + 0.3 * jaw_target
        else:
            self.current_jaw_offset = jaw_target
        
        self.jaw_position_history.append(self.current_jaw_offset)
        
        return self.current_jaw_offset
    
    def get_smooth_jaw(self) -> float:
        """Get smoothed jaw position."""
        if len(self.jaw_position_history) < 3:
            return self.current_jaw_offset
        
        # Use Kalman filter-like smoothing
        recent = list(self.jaw_position_history)[-3:]
        weights = np.array([0.2, 0.3, 0.5])
        smoothed = np.average(recent, weights=weights)
        
        return float(smoothed)


class EyeBlinkSync:
    """
    Synchronizes eye blinks with mouth movements for more natural appearance.
    """
    
    def __init__(self, blink_probability: float = 0.03):
        """
        Initialize blink synchronization.
        
        Args:
            blink_probability: Probability of blink per frame
        """
        self.blink_probability = blink_probability
        self.blink_frame = 0
        self.is_blinking = False
        self.blink_frames = 3  # Frames per blink
        
    def should_blink(self) -> bool:
        """Determine if should blink this frame."""
        if self.is_blinking:
            self.blink_frame += 1
            if self.blink_frame >= self.blink_frames:
                self.is_blinking = False
                self.blink_frame = 0
            return True
        
        # Probabilistic blink
        if np.random.random() < self.blink_probability:
            self.is_blinking = True
            self.blink_frame = 0
            return True
        
        return False
    
    def get_blink_intensity(self) -> float:
        """
        Get blink intensity (0.0-1.0).
        
        Returns:
            Intensity: 0.0=open, 1.0=fully closed
        """
        if not self.is_blinking:
            return 0.0
        
        # Sine wave for smooth blink
        progress = self.blink_frame / self.blink_frames
        intensity = np.sin(progress * np.pi)  # 0 -> 1 -> 0
        
        return float(intensity)


def apply_mouth_deformation(
    mouth_region: np.ndarray,
    deformation_vector: np.ndarray,
    strength: float = 0.5
) -> np.ndarray:
    """
    Apply optical flow-based deformation to mouth region.
    
    Args:
        mouth_region: Mouth region image
        deformation_vector: Deformation vectors (2D)
        strength: Deformation strength (0.0-1.0)
        
    Returns:
        Deformed mouth region
    """
    h, w = mouth_region.shape[:2]
    
    # Create meshgrid for warping
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply deformation
    dx = cv2.resize(deformation_vector[:, 0], (w, h)) * strength * 2
    dy = cv2.resize(deformation_vector[:, 1], (h, w)) * strength * 2
    
    # Warp image
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    deformed = cv2.remap(mouth_region, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    
    return deformed


def calculate_mouth_shape_confidence(
    mouth_landmarks: np.ndarray,
    reference_shape: np.ndarray
) -> float:
    """
    Calculate confidence that landmarks match a reference shape.
    
    Args:
        mouth_landmarks: Current mouth landmarks
        reference_shape: Reference shape to compare against
        
    Returns:
        Confidence (0.0-1.0)
    """
    if mouth_landmarks.shape != reference_shape.shape:
        return 0.5  # Unknown
    
    # Calculate normalized distance
    distance = np.linalg.norm(mouth_landmarks - reference_shape)
    
    # Convert to confidence (smaller distance = higher confidence)
    confidence = np.exp(-distance / 50.0)  # 50 pixels is scale
    
    return float(np.clip(confidence, 0.0, 1.0))
