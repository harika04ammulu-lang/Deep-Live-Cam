"""
Temporal Landmark Stabilization Module

Provides exponential moving average (EMA) smoothing for facial landmarks
to reduce jitter and improve naturalness of mouth geometry during face swap.

Key features:
- Per-frame landmark history tracking
- Exponential moving average with configurable smoothing factor
- Optional stronger smoothing for mouth-specific landmarks (48-68)
- Automatic buffer reset on face detection loss or confidence drop
- Minimal latency overhead
"""

import numpy as np
from typing import Optional, Dict, Tuple
from collections import deque
import threading

# Thread-safe lock for landmark buffer access
LANDMARK_SMOOTHER_LOCK = threading.Lock()

# Mouth landmark indices (0-indexed into 106-point set)
MOUTH_LANDMARKS = list(range(48, 68))  # Outer mouth contour
MOUTH_DETAIL_LANDMARKS = list(range(48, 77))  # Extended mouth region with inner contour

# Minimum confidence threshold to continue smoothing (0.0-1.0)
MIN_FACE_CONFIDENCE = 0.3


class LandmarkSmoothingBuffer:
    """
    Maintains temporal landmark history and applies EMA smoothing.
    
    Tracks landmarks per detected face (by approximate position) and smooths
    across frames to reduce jitter while preserving natural motion.
    """
    
    def __init__(
        self,
        history_size: int = 5,
        smoothing_factor: float = 0.7,
        mouth_smoothing_factor: Optional[float] = None,
        enable_debug: bool = False
    ):
        """
        Initialize landmark smoothing buffer.
        
        Args:
            history_size: Number of previous frames to track (3-5 recommended)
            smoothing_factor: EMA weight for general landmarks (0.0-1.0).
                             Higher = more smoothing, more lag.
                             0.7 recommended for balance.
            mouth_smoothing_factor: Optional stronger smoothing for mouth (48-68).
                                   If None, uses smoothing_factor.
                                   Range 0.8-0.95 recommended.
            enable_debug: Enable debug output for monitoring.
        """
        self.history_size = max(1, min(history_size, 10))
        self.smoothing_factor = max(0.0, min(smoothing_factor, 1.0))
        self.mouth_smoothing_factor = (
            max(0.0, min(mouth_smoothing_factor, 1.0))
            if mouth_smoothing_factor is not None
            else self.smoothing_factor
        )
        self.enable_debug = enable_debug
        
        # Per-face landmark history (keyed by face position hash)
        self.landmark_history: Dict[str, deque] = {}
        self.smoothed_landmarks_cache: Dict[str, np.ndarray] = {}
        self.confidence_history: Dict[str, deque] = {}
        
        self.stats = {
            'processed_frames': 0,
            'smoothing_applied': 0,
            'buffer_resets': 0,
        }
    
    def _get_face_id(self, bbox: np.ndarray) -> str:
        """Generate stable face ID from bounding box for history tracking."""
        if bbox is None or len(bbox) < 4:
            return "unknown"
        # Use bbox center for face identification (quantized to 10px grid for stability)
        center_x = int((bbox[0] + bbox[2]) / 2 / 10) * 10
        center_y = int((bbox[1] + bbox[3]) / 2 / 10) * 10
        return f"face_{center_x}_{center_y}"
    
    def _should_reset_buffer(
        self,
        face_id: str,
        current_confidence: float,
        landmarks: np.ndarray
    ) -> bool:
        """
        Determine if buffer should be reset based on confidence drop or face loss.
        
        Args:
            face_id: Identifier for the face
            current_confidence: Detection confidence (0.0-1.0)
            landmarks: Current landmarks
            
        Returns:
            True if buffer should be cleared
        """
        if current_confidence < MIN_FACE_CONFIDENCE:
            return True
        
        if face_id not in self.confidence_history:
            return False
        
        # Reset if confidence drops significantly (>0.2 drop)
        if self.confidence_history[face_id]:
            prev_conf = self.confidence_history[face_id][-1]
            if prev_conf - current_confidence > 0.2:
                return True
        
        return False
    
    def smooth(
        self,
        landmarks: np.ndarray,
        face_bbox: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        apply_mouth_emphasis: bool = True
    ) -> np.ndarray:
        """
        Apply temporal EMA smoothing to landmarks.
        
        Args:
            landmarks: 106-point landmarks (shape: 106x2)
            face_bbox: Bounding box for face identification [x1, y1, x2, y2]
            confidence: Detection confidence for this frame
            apply_mouth_emphasis: Apply stronger smoothing to mouth landmarks
            
        Returns:
            Smoothed landmarks (same shape as input)
        """
        with LANDMARK_SMOOTHER_LOCK:
            if landmarks is None or landmarks.shape[0] < 68:
                return landmarks
            
            landmarks = landmarks.astype(np.float32)
            face_id = self._get_face_id(face_bbox)
            
            # Reset buffer if confidence dropped or face was lost
            if self._should_reset_buffer(face_id, confidence, landmarks):
                if face_id in self.landmark_history:
                    del self.landmark_history[face_id]
                    del self.confidence_history[face_id]
                    if self.enable_debug:
                        print(f"[Landmark Smoother] Buffer reset for {face_id} (confidence: {confidence:.2f})")
                    self.stats['buffer_resets'] += 1
            
            # Initialize history for new face
            if face_id not in self.landmark_history:
                self.landmark_history[face_id] = deque(maxlen=self.history_size)
                self.confidence_history[face_id] = deque(maxlen=self.history_size)
            
            # Add current landmarks to history
            self.landmark_history[face_id].append(landmarks.copy())
            self.confidence_history[face_id].append(confidence)
            
            # Apply EMA smoothing
            smoothed = self._apply_ema(
                self.landmark_history[face_id],
                apply_mouth_emphasis
            )
            
            self.smoothed_landmarks_cache[face_id] = smoothed
            self.stats['processed_frames'] += 1
            self.stats['smoothing_applied'] += 1
            
            if self.enable_debug and self.stats['processed_frames'] % 30 == 0:
                print(f"[Landmark Smoother] Frame {self.stats['processed_frames']}: "
                      f"Applied smoothing to {face_id}")
            
            return smoothed
    
    def _apply_ema(
        self,
        landmark_history: deque,
        apply_mouth_emphasis: bool
    ) -> np.ndarray:
        """
        Apply exponential moving average across landmark history.
        
        Args:
            landmark_history: Deque of landmark arrays (each 106x2)
            apply_mouth_emphasis: Apply stronger smoothing to mouth indices
            
        Returns:
            EMA-smoothed landmarks
        """
        if len(landmark_history) == 0:
            return None
        
        if len(landmark_history) == 1:
            # Only one frame, return as-is
            return landmark_history[0].copy()
        
        # Stack all landmarks in history: (history_len, 106, 2)
        stacked = np.array(list(landmark_history))  # shape: (N, 106, 2)
        
        # Apply EMA with per-point weighting
        result = np.zeros_like(stacked[0])
        
        for point_idx in range(stacked.shape[1]):  # 106 points
            point_history = stacked[:, point_idx, :]  # (N, 2)
            
            # Determine smoothing factor for this point
            if apply_mouth_emphasis and point_idx in MOUTH_LANDMARKS:
                alpha = self.mouth_smoothing_factor
            else:
                alpha = self.smoothing_factor
            
            # EMA formula: smoothed = alpha * current + (1 - alpha) * previous_smoothed
            smoothed_point = point_history[0].copy()  # Start with oldest frame
            for i in range(1, len(point_history)):
                current = point_history[i]
                smoothed_point = alpha * current + (1.0 - alpha) * smoothed_point
            
            result[point_idx] = smoothed_point
        
        return result
    
    def get_smoothed_landmarks(self, face_id: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Retrieve cached smoothed landmarks for a face.
        
        Args:
            face_id: Face identifier (from _get_face_id())
            
        Returns:
            Smoothed landmarks or None if not available
        """
        with LANDMARK_SMOOTHER_LOCK:
            if face_id is None:
                # Return most recent
                if self.smoothed_landmarks_cache:
                    return list(self.smoothed_landmarks_cache.values())[-1]
                return None
            return self.smoothed_landmarks_cache.get(face_id)
    
    def reset_all(self):
        """Clear all landmark history buffers."""
        with LANDMARK_SMOOTHER_LOCK:
            self.landmark_history.clear()
            self.confidence_history.clear()
            self.smoothed_landmarks_cache.clear()
            if self.enable_debug:
                print("[Landmark Smoother] All buffers reset")
            self.stats['buffer_resets'] += 1
    
    def get_stats(self) -> Dict:
        """Get smoothing statistics."""
        with LANDMARK_SMOOTHER_LOCK:
            return self.stats.copy()


# Global singleton instance
_GLOBAL_SMOOTHER: Optional[LandmarkSmoothingBuffer] = None
_SMOOTHER_INIT_LOCK = threading.Lock()


def get_landmark_smoother(
    history_size: int = 5,
    smoothing_factor: float = 0.7,
    mouth_smoothing_factor: Optional[float] = 0.85,
    enable_debug: bool = False
) -> LandmarkSmoothingBuffer:
    """
    Get or create the global landmark smoother instance.
    
    Args:
        history_size: Frames to track (default: 5)
        smoothing_factor: General EMA factor (default: 0.7)
        mouth_smoothing_factor: Mouth EMA factor (default: 0.85)
        enable_debug: Enable debug output (default: False)
        
    Returns:
        Global LandmarkSmoothingBuffer instance
    """
    global _GLOBAL_SMOOTHER
    
    if _GLOBAL_SMOOTHER is None:
        with _SMOOTHER_INIT_LOCK:
            if _GLOBAL_SMOOTHER is None:
                _GLOBAL_SMOOTHER = LandmarkSmoothingBuffer(
                    history_size=history_size,
                    smoothing_factor=smoothing_factor,
                    mouth_smoothing_factor=mouth_smoothing_factor,
                    enable_debug=enable_debug
                )
    
    return _GLOBAL_SMOOTHER


def smooth_landmarks(
    landmarks: np.ndarray,
    face_bbox: Optional[np.ndarray] = None,
    confidence: float = 1.0,
    apply_mouth_emphasis: bool = True,
    smoothing_enabled: bool = True,
    smoothing_factor: float = 0.7,
    mouth_smoothing_factor: float = 0.85
) -> np.ndarray:
    """
    Convenience function to apply landmark smoothing.
    
    Args:
        landmarks: Input landmarks (106x2)
        face_bbox: Face bounding box for tracking
        confidence: Detection confidence
        apply_mouth_emphasis: Stronger smoothing for mouth
        smoothing_enabled: Whether to apply smoothing
        smoothing_factor: General smoothing strength
        mouth_smoothing_factor: Mouth smoothing strength
        
    Returns:
        Smoothed or original landmarks
    """
    if not smoothing_enabled or landmarks is None:
        return landmarks
    
    smoother = get_landmark_smoother(
        smoothing_factor=smoothing_factor,
        mouth_smoothing_factor=mouth_smoothing_factor
    )
    
    return smoother.smooth(
        landmarks,
        face_bbox=face_bbox,
        confidence=confidence,
        apply_mouth_emphasis=apply_mouth_emphasis
    )


def reset_landmark_smoother():
    """Reset the global landmark smoother."""
    global _GLOBAL_SMOOTHER
    if _GLOBAL_SMOOTHER is not None:
        _GLOBAL_SMOOTHER.reset_all()
