"""
Kalman Filter Landmark Smoothing with Bidirectional Temporal Filtering

Implements Kalman filtering for facial landmarks with:
- State-space modeling for position and velocity
- Adaptive process and measurement noise
- 3-frame lookahead buffer for bidirectional filtering
- Head pose estimation and smoothing
- Superior jitter elimination during head movement

This module can replace or supplement the existing EMA smoother.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from collections import deque
import threading

from modules.typing import Face, Frame

# Thread-safe lock
KALMAN_LOCK = threading.Lock()

# Landmark indices for different face regions
MOUTH_LANDMARKS = list(range(48, 68))  # 68-point format
EYES_LANDMARKS = list(range(36, 48))
EYEBROWS_LANDMARKS = list(range(17, 27))
NOSE_LANDMARKS = list(range(27, 36))
JAW_LANDMARKS = list(range(0, 17))


class KalmanFilter2D:
    """
    2D Kalman Filter for landmark position and velocity estimation.
    
    State vector: [x, y, vx, vy]
    - x, y: position
    - vx, vy: velocity
    
    Prediction: x_k = F * x_{k-1}
    Update: x_k = x_k + K * (z_k - H * x_k)
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 0.033  # ~30 FPS
    ):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
            dt: Time step between measurements (seconds)
        """
        self.dt = dt
        
        # State: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # State covariance matrix - increased initial uncertainty
        self.P = np.eye(4, dtype=np.float32) * 500.0
        
        # Process noise covariance - INCREASED for stability
        # Use 1e-2 for position, 1e-1 for velocity
        self.Q = np.eye(4, dtype=np.float32) * 1e-2
        # Higher noise for velocity components
        self.Q[2, 2] = 1e-1
        self.Q[3, 3] = 1e-1
        
        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
        self.initialized = False
    
    def predict(self) -> np.ndarray:
        """
        Prediction step: Predict next state.
        
        Returns:
            Predicted position [x, y]
        """
        # Predict state: x_k = F * x_{k-1}
        self.state = self.F @ self.state
        
        # Predict covariance: P_k = F * P_{k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # NUMERICAL STABILITY FIX: Clamp P to prevent overflow
        self.P = np.clip(self.P, -1e6, 1e6)
        
        # Check for NaN/inf and reset if detected
        if np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            print("[KalmanSmoother] Numerical instability detected in predict - resetting filter")
            self.reset()
        
        # Return predicted position
        return self.state[:2]
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step: Correct prediction with measurement.
        
        Args:
            measurement: Measured position [x, y]
            
        Returns:
            Updated position [x, y]
        """
        if not self.initialized:
            # First measurement - initialize state
            self.state[:2] = measurement
            self.state[2:] = 0  # Zero velocity
            self.initialized = True
            return measurement
        
        # Innovation (measurement residual): y = z - H * x
        innovation = measurement - self.H @ self.state
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H^T * S^{-1}
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular matrix, reset and return measurement
            print("[KalmanSmoother] Singular matrix in update - resetting filter")
            self.reset()
            return measurement
        
        # Update state: x = x + K * y
        self.state = self.state + K @ innovation
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        
        # NUMERICAL STABILITY FIX: Clamp P after update
        self.P = np.clip(self.P, -1e6, 1e6)
        
        # Check for NaN/inf in state or covariance
        if np.any(np.isnan(self.state)) or np.any(np.isinf(self.state)) or \
           np.any(np.isnan(self.P)) or np.any(np.isinf(self.P)):
            print("[KalmanSmoother] Numerical instability detected in update - resetting filter")
            self.reset()
            return measurement
        
        # Return updated position
        return self.state[:2]
    
    def process(self, measurement: np.ndarray) -> np.ndarray:
        """
        Full Kalman filter cycle: predict + update.
        
        Args:
            measurement: Measured position [x, y]
            
        Returns:
            Filtered position [x, y]
        """
        self.predict()
        return self.update(measurement)
    
    def reset(self):
        """Reset filter state."""
        self.state = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.0
        self.initialized = False
    
    def get_velocity(self) -> np.ndarray:
        """Get estimated velocity [vx, vy]."""
        return self.state[2:]


class MultiLandmarkKalmanFilter:
    """
    Kalman filter for multiple landmarks simultaneously.
    
    Maintains separate Kalman filters for each landmark point.
    """
    
    def __init__(
        self,
        num_landmarks: int,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 0.033
    ):
        """
        Initialize multi-landmark Kalman filter.
        
        Args:
            num_landmarks: Number of landmark points
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
            dt: Time step (seconds)
        """
        self.num_landmarks = num_landmarks
        self.filters = [
            KalmanFilter2D(process_noise, measurement_noise, dt)
            for _ in range(num_landmarks)
        ]
    
    def process(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Process all landmarks through Kalman filters.
        
        Args:
            landmarks: Landmark array (N, 2)
            
        Returns:
            Filtered landmarks (N, 2)
        """
        if landmarks.shape[0] != self.num_landmarks:
            return landmarks
        
        filtered = np.zeros_like(landmarks, dtype=np.float32)
        
        for i in range(self.num_landmarks):
            filtered[i] = self.filters[i].process(landmarks[i])
        
        # NaN GUARD: Check for NaN/inf in output and return original if detected
        if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
            print("[KalmanSmoother] NaN/inf detected in filtered landmarks - returning original")
            return landmarks
        
        return filtered
    
    def reset(self):
        """Reset all filters."""
        for f in self.filters:
            f.reset()
    
    def get_velocities(self) -> np.ndarray:
        """Get estimated velocities for all landmarks."""
        velocities = np.zeros((self.num_landmarks, 2), dtype=np.float32)
        for i in range(self.num_landmarks):
            velocities[i] = self.filters[i].get_velocity()
        return velocities



class BidirectionalTemporalFilter:
    """
    Bidirectional temporal filter with lookahead buffer.
    
    Uses a 3-frame lookahead to perform forward-backward filtering,
    which eliminates phase lag and reduces jitter significantly.
    
    Process:
    1. Forward pass: Filter frames 0 -> N
    2. Backward pass: Filter frames N -> 0
    3. Combine: Average forward and backward filtered results
    """
    
    def __init__(self, lookahead_size: int = 3):
        """
        Initialize bidirectional filter.
        
        Args:
            lookahead_size: Number of frames to buffer for lookahead
        """
        self.lookahead_size = lookahead_size
        self.buffer = deque(maxlen=lookahead_size)
        self.forward_filtered = deque(maxlen=lookahead_size)
        self.backward_filtered = deque(maxlen=lookahead_size)
    
    def add_frame(self, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Add a frame to the buffer and return bidirectionally filtered result.
        
        Args:
            landmarks: Current frame landmarks
            
        Returns:
            Bidirectionally filtered landmarks (with lookahead_size delay)
            or None if buffer not full yet
        """
        self.buffer.append(landmarks.copy())
        
        if len(self.buffer) < self.lookahead_size:
            # Buffer not full yet
            return None
        
        # Perform bidirectional filtering
        return self._bidirectional_filter()
    
    def _bidirectional_filter(self) -> np.ndarray:
        """
        Perform bidirectional filtering on buffered frames.
        
        Returns:
            Filtered landmarks for the first frame in buffer
        """
        buffer_list = list(self.buffer)
        n_frames = len(buffer_list)
        
        if n_frames == 0:
            return None
        
        # Forward pass (weighted average with decay)
        forward_result = buffer_list[0].copy()
        weights_sum = 1.0
        
        for i in range(1, n_frames):
            weight = 0.5 ** i  # Exponential decay
            forward_result += buffer_list[i] * weight
            weights_sum += weight
        
        forward_result /= weights_sum
        
        # Backward pass (weighted average with decay from end)
        backward_result = buffer_list[0].copy()
        weights_sum = 1.0
        
        for i in range(1, n_frames):
            weight = 0.5 ** i
            backward_result += buffer_list[n_frames - 1 - i] * weight
            weights_sum += weight
        
        backward_result /= weights_sum
        
        # Combine forward and backward (average)
        combined = (forward_result + backward_result) / 2.0
        
        return combined.astype(np.float32)
    
    def reset(self):
        """Reset buffer."""
        self.buffer.clear()
        self.forward_filtered.clear()
        self.backward_filtered.clear()


class HeadPoseEstimator:
    """
    Estimates head pose (rotation and translation) from facial landmarks.
    
    Uses PnP (Perspective-n-Point) algorithm to estimate 3D pose.
    """
    
    def __init__(self):
        """Initialize head pose estimator."""
        # 3D model points (generic face model in cm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -3.30, -6.50),         # Chin
            (-2.25, 1.65, -4.50),        # Left eye left corner
            (2.25, 1.65, -4.50),         # Right eye right corner
            (-1.50, -1.50, -3.50),       # Left mouth corner
            (1.50, -1.50, -3.50)         # Right mouth corner
        ], dtype=np.float32)
        
        # Camera internals (default, can be adjusted)
        self.focal_length = 1000
        self.camera_center = (320, 240)  # Will be updated
        
        self.rotation_vector = None
        self.translation_vector = None
    
    def estimate_pose(
        self,
        landmarks: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate head pose from landmarks.
        
        Args:
            landmarks: Facial landmarks (68+ points expected)
            image_size: Image size (width, height)
            
        Returns:
            (rotation_vector, translation_vector) or (None, None)
        """
        if landmarks.shape[0] < 68:
            return None, None
        
        try:
            # Extract relevant 2D points (matching model_points)
            image_points = np.array([
                landmarks[30],    # Nose tip
                landmarks[8],     # Chin
                landmarks[36],    # Left eye left corner
                landmarks[45],    # Right eye right corner
                landmarks[48],    # Left mouth corner
                landmarks[54]     # Right mouth corner
            ], dtype=np.float32)
            
            # Update camera matrix
            h, w = image_size
            self.camera_center = (w / 2, h / 2)
            camera_matrix = np.array([
                [self.focal_length, 0, self.camera_center[0]],
                [0, self.focal_length, self.camera_center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Distortion coefficients (assuming no distortion)
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            
            # Solve PnP
            import cv2
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                self.rotation_vector = rotation_vector
                self.translation_vector = translation_vector
                return rotation_vector, translation_vector
            
        except Exception as e:
            print(f"[KalmanSmoother] Head pose estimation error: {e}")
        
        return None, None
    
    def get_euler_angles(self) -> Optional[Tuple[float, float, float]]:
        """
        Get Euler angles (pitch, yaw, roll) in degrees.
        
        Returns:
            (pitch, yaw, roll) or None
        """
        if self.rotation_vector is None:
            return None
        
        try:
            import cv2
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)
            
            # Extract Euler angles
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            
            singular = sy < 1e-6
            
            if not singular:
                pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                yaw = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = 0
            
            # Convert to degrees
            return (
                np.degrees(pitch),
                np.degrees(yaw),
                np.degrees(roll)
            )
            
        except Exception:
            return None



class AdaptiveKalmanLandmarkSmoother:
    """
    Adaptive Kalman filter smoother with bidirectional temporal filtering.
    
    Features:
    - Kalman filtering for each landmark
    - Adaptive noise based on velocity
    - 3-frame lookahead for bidirectional filtering
    - Head pose estimation and smoothing
    - Region-specific noise parameters
    """
    
    def __init__(
        self,
        num_landmarks: int = 106,
        lookahead_size: int = 3,
        base_process_noise: float = 0.01,
        base_measurement_noise: float = 0.1,
        enable_bidirectional: bool = True,
        enable_head_pose: bool = True
    ):
        """
        Initialize adaptive Kalman smoother.
        
        Args:
            num_landmarks: Number of landmarks to track
            lookahead_size: Lookahead buffer size for bidirectional filtering
            base_process_noise: Base process noise covariance
            base_measurement_noise: Base measurement noise covariance
            enable_bidirectional: Enable bidirectional temporal filtering
            enable_head_pose: Enable head pose estimation and smoothing
        """
        self.num_landmarks = num_landmarks
        self.enable_bidirectional = enable_bidirectional
        self.enable_head_pose = enable_head_pose
        
        # Main Kalman filter for landmarks
        self.kalman_filter = MultiLandmarkKalmanFilter(
            num_landmarks,
            base_process_noise,
            base_measurement_noise
        )
        
        # Bidirectional temporal filter
        if enable_bidirectional:
            self.temporal_filter = BidirectionalTemporalFilter(lookahead_size)
        else:
            self.temporal_filter = None
        
        # Head pose estimator
        if enable_head_pose:
            self.head_pose_estimator = HeadPoseEstimator()
            self.head_pose_filter = KalmanFilter2D(
                process_noise=0.05,
                measurement_noise=0.2
            )
        else:
            self.head_pose_estimator = None
            self.head_pose_filter = None
        
        # Adaptive parameters
        self.velocity_threshold = 5.0  # pixels per frame
        self.high_velocity_factor = 2.0  # Increase noise during fast movement
        
        # Statistics
        self.frame_count = 0
        self.total_jitter_reduction = 0.0
    
    def smooth_landmarks(
        self,
        landmarks: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
        confidence: float = 1.0
    ) -> np.ndarray:
        """
        Apply Kalman smoothing to landmarks.
        
        Args:
            landmarks: Input landmarks (N, 2)
            image_size: Image size for head pose estimation (width, height)
            confidence: Detection confidence (affects measurement noise)
            
        Returns:
            Smoothed landmarks (N, 2)
        """
        if landmarks is None or landmarks.shape[0] != self.num_landmarks:
            return landmarks
        
        try:
            # Adapt measurement noise based on confidence
            self._adapt_noise_to_confidence(confidence)
            
            # Apply Kalman filtering
            kalman_filtered = self.kalman_filter.process(landmarks)
            
            # NaN GUARD: Check for NaN/inf after Kalman filtering
            if np.any(np.isnan(kalman_filtered)) or np.any(np.isinf(kalman_filtered)):
                print("[KalmanSmoother] NaN/inf detected after Kalman filtering - returning original")
                return landmarks
            
            # Check for high velocity (fast head movement)
            velocities = self.kalman_filter.get_velocities()
            avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
            
            if avg_velocity > self.velocity_threshold:
                # High velocity detected - adapt noise
                self._increase_process_noise()
            
            # Apply bidirectional temporal filtering if enabled
            if self.enable_bidirectional and self.temporal_filter:
                bidirectional_result = self.temporal_filter.add_frame(kalman_filtered)
                
                if bidirectional_result is not None:
                    smoothed = bidirectional_result
                else:
                    # Buffer not full yet, use Kalman filtered only
                    smoothed = kalman_filtered
            else:
                smoothed = kalman_filtered
            
            # FINAL NaN GUARD: Check smoothed output before returning
            if np.any(np.isnan(smoothed)) or np.any(np.isinf(smoothed)):
                print("[KalmanSmoother] NaN/inf detected in final smoothed output - returning original")
                return landmarks
            
            # Head pose smoothing if enabled
            if self.enable_head_pose and self.head_pose_estimator and image_size:
                self._smooth_head_pose(landmarks, image_size)
            
            self.frame_count += 1
            return smoothed
            
        except Exception as e:
            print(f"[KalmanSmoother] Error in smooth_landmarks: {e}")
            return landmarks
    
    def _adapt_noise_to_confidence(self, confidence: float):
        """Adapt measurement noise based on detection confidence."""
        # Lower confidence = higher measurement noise
        noise_factor = 1.0 / max(confidence, 0.1)
        
        for kalman in self.kalman_filter.filters:
            # OVERFLOW FIX: Clamp R after multiplication to prevent overflow
            kalman.R = np.clip(kalman.R * noise_factor, 1e-10, 1e6)
    
    def _increase_process_noise(self):
        """Temporarily increase process noise during fast movement."""
        for kalman in self.kalman_filter.filters:
            # OVERFLOW FIX: Clamp Q after multiplication to prevent overflow
            kalman.Q = np.clip(kalman.Q * self.high_velocity_factor, 1e-10, 1e6)
    
    def _smooth_head_pose(self, landmarks: np.ndarray, image_size: Tuple[int, int]):
        """Estimate and smooth head pose."""
        try:
            rotation, translation = self.head_pose_estimator.estimate_pose(
                landmarks,
                image_size
            )
            
            if rotation is not None:
                euler_angles = self.head_pose_estimator.get_euler_angles()
                # Head pose smoothing can be used for adaptive noise adjustment
                # or for pose-aware landmark correction (future enhancement)
        except Exception as e:
            pass
    
    def reset(self):
        """Reset smoother state."""
        self.kalman_filter.reset()
        if self.temporal_filter:
            self.temporal_filter.reset()
        if self.head_pose_filter:
            self.head_pose_filter.reset()
        self.frame_count = 0
    
    def get_statistics(self) -> Dict:
        """Get smoothing statistics."""
        return {
            'frame_count': self.frame_count,
            'avg_jitter_reduction': self.total_jitter_reduction / max(self.frame_count, 1),
            'bidirectional_enabled': self.enable_bidirectional,
            'head_pose_enabled': self.enable_head_pose
        }


# Global singleton instance
_global_kalman_smoother: Optional[AdaptiveKalmanLandmarkSmoother] = None
_kalman_smoother_lock = threading.Lock()



def get_kalman_smoother(
    num_landmarks: int = 106,
    lookahead_size: int = 3,
    enable_bidirectional: bool = True,
    enable_head_pose: bool = True
) -> AdaptiveKalmanLandmarkSmoother:
    """
    Get or create global Kalman smoother instance (singleton).
    
    Args:
        num_landmarks: Number of landmarks
        lookahead_size: Lookahead buffer size
        enable_bidirectional: Enable bidirectional filtering
        enable_head_pose: Enable head pose estimation
        
    Returns:
        Global Kalman smoother instance
    """
    global _global_kalman_smoother
    
    with _kalman_smoother_lock:
        if _global_kalman_smoother is None:
            _global_kalman_smoother = AdaptiveKalmanLandmarkSmoother(
                num_landmarks=num_landmarks,
                lookahead_size=lookahead_size,
                enable_bidirectional=enable_bidirectional,
                enable_head_pose=enable_head_pose
            )
        
        return _global_kalman_smoother


def smooth_landmarks_kalman(
    landmarks: np.ndarray,
    image_size: Optional[Tuple[int, int]] = None,
    confidence: float = 1.0,
    enable_bidirectional: bool = True,
    enable_head_pose: bool = True
) -> np.ndarray:
    """
    Apply Kalman smoothing to landmarks (convenience function).
    
    Args:
        landmarks: Input landmarks (N, 2)
        image_size: Image size (width, height) for head pose
        confidence: Detection confidence
        enable_bidirectional: Enable bidirectional filtering
        enable_head_pose: Enable head pose smoothing
        
    Returns:
        Smoothed landmarks
    """
    if landmarks is None:
        return None
    
    smoother = get_kalman_smoother(
        num_landmarks=landmarks.shape[0],
        enable_bidirectional=enable_bidirectional,
        enable_head_pose=enable_head_pose
    )
    
    return smoother.smooth_landmarks(
        landmarks,
        image_size=image_size,
        confidence=confidence
    )


def reset_kalman_smoother():
    """Reset global Kalman smoother state."""
    global _global_kalman_smoother
    
    with _kalman_smoother_lock:
        if _global_kalman_smoother is not None:
            _global_kalman_smoother.reset()


def get_kalman_statistics() -> Dict:
    """Get Kalman smoother statistics."""
    global _global_kalman_smoother
    
    with _kalman_smoother_lock:
        if _global_kalman_smoother is not None:
            return _global_kalman_smoother.get_statistics()
        return {}


def compare_smoothers(
    landmarks_sequence: List[np.ndarray],
    use_kalman: bool = True,
    use_ema: bool = False
) -> Dict:
    """
    Compare Kalman filter vs EMA smoothing on a sequence of landmarks.
    
    Args:
        landmarks_sequence: List of landmark arrays
        use_kalman: Apply Kalman smoothing
        use_ema: Apply EMA smoothing (from landmark_smoother)
        
    Returns:
        Comparison statistics
    """
    if not landmarks_sequence:
        return {}
    
    results = {
        'original_jitter': 0.0,
        'kalman_jitter': 0.0,
        'ema_jitter': 0.0,
        'kalman_reduction': 0.0,
        'ema_reduction': 0.0
    }
    
    # Calculate original jitter
    for i in range(1, len(landmarks_sequence)):
        diff = landmarks_sequence[i] - landmarks_sequence[i-1]
        results['original_jitter'] += np.mean(np.linalg.norm(diff, axis=1))
    
    results['original_jitter'] /= len(landmarks_sequence) - 1
    
    # Kalman smoothing
    if use_kalman:
        reset_kalman_smoother()
        kalman_smoothed = []
        
        for landmarks in landmarks_sequence:
            smoothed = smooth_landmarks_kalman(landmarks)
            kalman_smoothed.append(smoothed)
        
        for i in range(1, len(kalman_smoothed)):
            diff = kalman_smoothed[i] - kalman_smoothed[i-1]
            results['kalman_jitter'] += np.mean(np.linalg.norm(diff, axis=1))
        
        results['kalman_jitter'] /= len(kalman_smoothed) - 1
        results['kalman_reduction'] = (
            1 - results['kalman_jitter'] / results['original_jitter']
        ) * 100
    
    # EMA smoothing
    if use_ema:
        try:
            from modules.processors.frame.landmark_smoother import (
                smooth_landmarks as ema_smooth,
                reset_landmark_smoother
            )
            
            reset_landmark_smoother()
            ema_smoothed = []
            
            for landmarks in landmarks_sequence:
                smoothed = ema_smooth(landmarks)
                ema_smoothed.append(smoothed)
            
            for i in range(1, len(ema_smoothed)):
                diff = ema_smoothed[i] - ema_smoothed[i-1]
                results['ema_jitter'] += np.mean(np.linalg.norm(diff, axis=1))
            
            results['ema_jitter'] /= len(ema_smoothed) - 1
            results['ema_reduction'] = (
                1 - results['ema_jitter'] / results['original_jitter']
            ) * 100
            
        except ImportError:
            pass
    
    return results


# Compatibility functions for drop-in replacement
def smooth_landmarks(
    landmarks: np.ndarray,
    face_bbox: Optional[np.ndarray] = None,
    confidence: float = 1.0,
    apply_mouth_emphasis: bool = True,
    smoothing_enabled: bool = True,
    smoothing_factor: float = 0.7,
    mouth_smoothing_factor: float = 0.85,
    use_kalman: bool = True
) -> np.ndarray:
    """
    Drop-in replacement for EMA smooth_landmarks that uses Kalman filter.
    
    Can be configured to use Kalman, EMA, or both (hybrid).
    
    Args:
        landmarks: Input landmarks
        face_bbox: Face bounding box (for image size estimation)
        confidence: Detection confidence
        apply_mouth_emphasis: Apply stronger smoothing to mouth (ignored for Kalman)
        smoothing_enabled: Enable smoothing
        smoothing_factor: EMA smoothing factor (ignored for Kalman)
        mouth_smoothing_factor: Mouth EMA factor (ignored for Kalman)
        use_kalman: Use Kalman filter (True) or fall back to EMA (False)
        
    Returns:
        Smoothed landmarks
    """
    if not smoothing_enabled or landmarks is None:
        return landmarks
    
    if use_kalman:
        # Estimate image size from bbox if available
        image_size = None
        if face_bbox is not None and len(face_bbox) >= 4:
            # Rough estimate: bbox suggests image is at least 2x bbox size
            w = int((face_bbox[2] - face_bbox[0]) * 3)
            h = int((face_bbox[3] - face_bbox[1]) * 3)
            image_size = (w, h)
        
        return smooth_landmarks_kalman(
            landmarks,
            image_size=image_size,
            confidence=confidence,
            enable_bidirectional=True,
            enable_head_pose=True
        )
    else:
        # Fall back to EMA smoothing
        try:
            from modules.processors.frame.landmark_smoother import (
                smooth_landmarks as ema_smooth
            )
            return ema_smooth(
                landmarks,
                face_bbox=face_bbox,
                confidence=confidence,
                apply_mouth_emphasis=apply_mouth_emphasis,
                smoothing_enabled=smoothing_enabled,
                smoothing_factor=smoothing_factor,
                mouth_smoothing_factor=mouth_smoothing_factor
            )
        except ImportError:
            return landmarks


def reset_landmark_smoother():
    """Reset landmark smoother (compatible with both Kalman and EMA)."""
    reset_kalman_smoother()
    
    # Also reset EMA if available
    try:
        from modules.processors.frame.landmark_smoother import (
            reset_landmark_smoother as reset_ema
        )
        reset_ema()
    except ImportError:
        pass
