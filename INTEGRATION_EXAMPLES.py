"""
Landmark Smoothing Integration Examples
========================================

This file shows how to use and integrate the temporal landmark smoothing
feature in your code.
"""

# ============================================================================
# EXAMPLE 1: Basic Usage in Live Processing
# ============================================================================

import modules.globals
from modules.processors.frame.landmark_smoother import smooth_landmarks
import cv2
import numpy as np

def process_live_frame(frame):
    """Process a webcam frame with landmark smoothing enabled."""
    
    # Enable smoothing (default is True, but showing for clarity)
    modules.globals.enable_landmark_smoothing = True
    
    # Your face detection code here
    detected_faces = detect_faces(frame)
    
    for target_face in detected_faces:
        # Landmarks are smoothed automatically in face_swapper.swap_face()
        # No explicit smoothing call needed!
        result_frame = swap_face(source_face, target_face, frame)
    
    return result_frame


# ============================================================================
# EXAMPLE 2: Manual Landmark Smoothing (Advanced)
# ============================================================================

from modules.processors.frame.landmark_smoother import (
    get_landmark_smoother,
    smooth_landmarks,
    reset_landmark_smoother,
    LandmarkSmoothingBuffer
)

def manually_smooth_landmarks(raw_landmarks, face_bbox, confidence):
    """
    Manually apply smoothing if you need more control.
    
    Typically not needed - the pipeline handles this automatically.
    But useful for custom processing flows.
    """
    
    # Apply smoothing using the global singleton
    smoothed = smooth_landmarks(
        landmarks=raw_landmarks,
        face_bbox=face_bbox,
        confidence=confidence,
        apply_mouth_emphasis=True,
        smoothing_enabled=True,
        smoothing_factor=0.7,
        mouth_smoothing_factor=0.85
    )
    
    return smoothed


# ============================================================================
# EXAMPLE 3: Getting Statistics and Monitoring
# ============================================================================

def print_smoothing_stats():
    """Print landmark smoothing statistics."""
    
    smoother = get_landmark_smoother()
    stats = smoother.get_stats()
    
    print(f"Landmark Smoothing Statistics:")
    print(f"  Frames processed: {stats['processed_frames']}")
    print(f"  Smoothing applied: {stats['smoothing_applied']}")
    print(f"  Buffer resets: {stats['buffer_resets']}")


# ============================================================================
# EXAMPLE 4: Debug Overlay Visualization
# ============================================================================

def enable_debug_visualization():
    """Enable landmark debug overlay to see raw vs smoothed."""
    
    modules.globals.show_landmark_debug_overlay = True
    
    # Now when frames are processed, they'll show:
    # - Green circles: raw detected landmarks
    # - Blue circles: smoothed landmarks
    # - Yellow crosses: mouth landmark positions
    # - Legend in top-left


# ============================================================================
# EXAMPLE 5: Custom Configuration for Different Use Cases
# ============================================================================

def configure_for_live_streaming():
    """Optimize settings for real-time streaming (responsiveness priority)."""
    modules.globals.enable_landmark_smoothing = True
    modules.globals.landmark_smoothing_factor = 0.65
    modules.globals.mouth_smoothing_factor = 0.80
    modules.globals.smoothing_history_size = 3  # Less smoothing
    print("Configured for LIVE STREAMING (responsive)")


def configure_for_recorded_video():
    """Optimize settings for recorded video (stability priority)."""
    modules.globals.enable_landmark_smoothing = True
    modules.globals.landmark_smoothing_factor = 0.75
    modules.globals.mouth_smoothing_factor = 0.90
    modules.globals.smoothing_history_size = 5  # More smoothing
    print("Configured for RECORDED VIDEO (stable)")


def configure_for_high_fps():
    """Optimize settings for 60+ fps streams."""
    modules.globals.enable_landmark_smoothing = True
    modules.globals.landmark_smoothing_factor = 0.80
    modules.globals.mouth_smoothing_factor = 0.92
    modules.globals.smoothing_history_size = 5
    print("Configured for HIGH FPS (60+ fps)")


# ============================================================================
# EXAMPLE 6: Reset Smoothing (Face Loss, Camera Switch, etc.)
# ============================================================================

def handle_face_loss():
    """Reset smoothing when face is no longer detected."""
    
    from modules.processors.frame.landmark_smoother import reset_landmark_smoother
    
    # This is called automatically when:
    # - Face confidence drops below threshold
    # - Face detection fails
    # - Switching cameras in live mode
    
    # You can also call it manually:
    reset_landmark_smoother()
    print("Landmark smoother reset")


def handle_video_transition():
    """Reset when transitioning between videos."""
    
    # When starting to process a new video file,
    # reset the landmark smoother to avoid carryover
    reset_landmark_smoother()
    print("Starting new video - landmark history cleared")


# ============================================================================
# EXAMPLE 7: Performance Monitoring
# ============================================================================

import time

def benchmark_smoothing_overhead():
    """Measure the performance impact of landmark smoothing."""
    
    import numpy as np
    from modules.processors.frame.landmark_smoother import smooth_landmarks
    
    # Create dummy landmarks
    landmarks = np.random.randn(106, 2).astype(np.float32) * 50 + 200
    bbox = np.array([100, 100, 300, 300])
    
    # Warmup
    for _ in range(10):
        smooth_landmarks(landmarks, face_bbox=bbox)
    
    # Benchmark
    num_iterations = 1000
    start = time.time()
    
    for _ in range(num_iterations):
        smooth_landmarks(landmarks, face_bbox=bbox)
    
    elapsed = time.time() - start
    avg_ms = (elapsed / num_iterations) * 1000
    
    print(f"Smoothing overhead: {avg_ms:.2f} ms per frame")
    print(f"Total for 30 fps: {avg_ms * 30:.2f} ms per second")
    print(f"Pipeline impact: {(avg_ms / 33.33) * 100:.1f}% at 30fps")


# ============================================================================
# EXAMPLE 8: Conditional Smoothing Based on Motion
# ============================================================================

def adaptive_smoothing(prev_landmarks, curr_landmarks, face_confidence):
    """
    Apply stronger smoothing when there's high motion (speech),
    lighter smoothing when face is still.
    
    This is an advanced technique for even better results.
    """
    
    # Calculate landmark motion magnitude
    motion = np.linalg.norm(curr_landmarks - prev_landmarks)
    
    if motion > 20:  # High motion (talking, animated)
        factor = 0.75  # Stronger smoothing
    elif motion > 10:  # Medium motion
        factor = 0.70  # Normal smoothing
    else:  # Low motion (still face)
        factor = 0.60  # Lighter smoothing (more responsive)
    
    # Apply custom smoothing
    smoothed = smooth_landmarks(
        curr_landmarks,
        smoothing_factor=factor,
        mouth_smoothing_factor=factor + 0.15
    )
    
    return smoothed


# ============================================================================
# EXAMPLE 9: Integration with Mouth Masking
# ============================================================================

def enable_with_mouth_mask():
    """Enable landmark smoothing alongside mouth masking for best results."""
    
    # Landmark smoothing improves geometry stability
    modules.globals.enable_landmark_smoothing = True
    
    # Mouth masking preserves original mouth motion
    # Together they create very natural results
    modules.globals.mouth_mask = True
    
    print("Enabled: Landmark smoothing + Mouth masking")
    print("Expected: Ultra-natural lip sync with preserved jaw motion")


# ============================================================================
# EXAMPLE 10: Per-Frame Processing Loop
# ============================================================================

def process_video_with_smoothing(input_video_path, output_video_path):
    """Complete example of processing a video with smoothing."""
    
    import cv2
    from modules.processors.frame.face_swapper import process_frames
    
    # Reset smoothing at start of video
    reset_landmark_smoother()
    
    # Configure for video quality
    configure_for_recorded_video()
    
    # Enable debug if needed
    modules.globals.show_landmark_debug_overlay = False
    
    # Your video processing loop
    cap = cv2.VideoCapture(input_video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with automatic landmark smoothing
        # (happens inside swap_face -> face_swapper.get)
        result = process_frame(source_face, frame)
        
        # Write result
        # ... your writing code here ...
    
    cap.release()
    
    # Print final statistics
    print_smoothing_stats()


# ============================================================================
# EXAMPLE 11: Testing Different Smoothing Factors
# ============================================================================

def test_smoothing_factors():
    """Test different smoothing factor combinations."""
    
    import numpy as np
    from modules.processors.frame.landmark_smoother import LandmarkSmoothingBuffer
    
    factors_to_test = [
        (0.60, 0.75),  # Light smoothing
        (0.70, 0.85),  # Balanced (default)
        (0.80, 0.95),  # Heavy smoothing
    ]
    
    landmarks = np.random.randn(106, 2).astype(np.float32) * 50 + 200
    
    for general, mouth in factors_to_test:
        smoother = LandmarkSmoothingBuffer(
            history_size=5,
            smoothing_factor=general,
            mouth_smoothing_factor=mouth,
            enable_debug=False
        )
        
        # Process some frames
        for i in range(30):
            noise = np.random.randn(106, 2).astype(np.float32) * 3
            smoothed = smoother.smooth(landmarks + noise)
        
        print(f"Factor ({general}, {mouth}): OK")


# ============================================================================
# USAGE QUICK REFERENCE
# ============================================================================

"""
AUTOMATIC (Default - No Code Needed):
  - Smoothing enabled by default
  - All configuration in modules.globals
  - Just call swap_face() normally

MANUAL (If You Need Custom Logic):
  from modules.processors.frame.landmark_smoother import (
      smooth_landmarks,
      reset_landmark_smoother,
      get_landmark_smoother
  )
  
  # Apply smoothing
  smoothed = smooth_landmarks(landmarks, face_bbox, confidence)
  
  # Reset buffer
  reset_landmark_smoother()
  
  # Get stats
  smoother = get_landmark_smoother()
  stats = smoother.get_stats()

CONFIGURATION:
  modules.globals.enable_landmark_smoothing = True/False
  modules.globals.landmark_smoothing_factor = 0.0-1.0  (0.7 default)
  modules.globals.mouth_smoothing_factor = 0.0-1.0     (0.85 default)
  modules.globals.smoothing_history_size = 1-10        (5 default)
  modules.globals.show_landmark_debug_overlay = True/False

MONITORING:
  smoother = get_landmark_smoother()
  stats = smoother.get_stats()
  # Returns: {
  #   'processed_frames': int,
  #   'smoothing_applied': int,
  #   'buffer_resets': int
  # }
"""
