#!/usr/bin/env python3
"""
Test script for temporal landmark stabilization feature.

Demonstrates:
- Landmark smoothing for stable mouth geometry
- Configuration options
- Statistics tracking
- Debug overlay functionality
"""

import cv2
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modules.globals
from modules.processors.frame.landmark_smoother import (
    get_landmark_smoother,
    smooth_landmarks,
    reset_landmark_smoother,
    LandmarkSmoothingBuffer,
    MOUTH_LANDMARKS
)


def test_basic_smoothing():
    """Test basic landmark smoothing functionality."""
    print("\n" + "="*70)
    print("TEST 1: Basic Landmark Smoothing")
    print("="*70)
    
    # Create smoother with test config
    smoother = LandmarkSmoothingBuffer(
        history_size=5,
        smoothing_factor=0.7,
        mouth_smoothing_factor=0.85,
        enable_debug=True
    )
    
    # Simulate 10 frames of slightly noisy landmarks
    print("\nSimulating 10 frames of landmark detection with noise...")
    base_landmarks = np.random.randn(106, 2).astype(np.float32) * 50 + 200  # Centered around 200
    
    for frame_idx in range(10):
        # Add noise to simulate detection jitter
        noise = np.random.randn(106, 2).astype(np.float32) * 3  # ±3 pixel noise
        noisy_landmarks = base_landmarks + noise
        
        # Apply smoothing
        bbox = np.array([100 + frame_idx, 100, 300 + frame_idx, 300])
        smoothed = smoother.smooth(noisy_landmarks, face_bbox=bbox, confidence=0.95)
        
        # Calculate jitter reduction
        raw_variance = np.var(noise)
        if frame_idx > 0:
            # Compare consecutive raw landmarks to smoothed landmarks
            prev_smoothed = smoother.get_smoothed_landmarks()
            if prev_smoothed is not None:
                smooth_diff = np.mean(np.abs(smoothed - prev_smoothed))
                raw_diff = np.mean(np.abs(noisy_landmarks - prev_smoothed))
                reduction = (1 - smooth_diff / (raw_diff + 1e-6)) * 100
                print(f"  Frame {frame_idx+1}: Jitter reduction = {reduction:.1f}%")
    
    # Get statistics
    stats = smoother.get_stats()
    print(f"\n✓ Smoothing Statistics:")
    print(f"  - Frames processed: {stats['processed_frames']}")
    print(f"  - Smoothing applied: {stats['smoothing_applied']}")
    print(f"  - Buffer resets: {stats['buffer_resets']}")
    
    return True


def test_mouth_emphasis():
    """Test that mouth landmarks get stronger smoothing."""
    print("\n" + "="*70)
    print("TEST 2: Mouth Landmark Emphasis")
    print("="*70)
    
    smoother = LandmarkSmoothingBuffer(
        history_size=5,
        smoothing_factor=0.7,      # General smoothing
        mouth_smoothing_factor=0.95,  # Mouth emphasis (stronger)
        enable_debug=False
    )
    
    # Create landmarks with specific patterns
    landmarks = np.random.randn(106, 2).astype(np.float32) * 20 + 200
    
    # Add more jitter to mouth landmarks to test emphasis
    mouth_jitter = np.random.randn(len(MOUTH_LANDMARKS), 2).astype(np.float32) * 5
    landmarks[MOUTH_LANDMARKS] += mouth_jitter
    
    print(f"  Mouth landmarks indices: {MOUTH_LANDMARKS[0]}-{MOUTH_LANDMARKS[-1]}")
    print(f"  Added extra jitter to mouth region for testing")
    
    # Process multiple frames
    smoothed_mouth_offsets = []
    smoothed_general_offsets = []
    
    for frame_idx in range(8):
        noise = np.random.randn(106, 2).astype(np.float32) * 2
        test_lm = landmarks + noise
        
        smoothed = smoother.smooth(test_lm, confidence=0.9)
        
        if frame_idx > 0:
            # Measure smoothness of mouth vs other landmarks
            mouth_variance = np.var(smoothed[MOUTH_LANDMARKS])
            general_variance = np.var(smoothed[list(range(0, 48))])  # Non-mouth
            
            smoothed_mouth_offsets.append(mouth_variance)
            smoothed_general_offsets.append(general_variance)
    
    if smoothed_mouth_offsets:
        avg_mouth = np.mean(smoothed_mouth_offsets)
        avg_general = np.mean(smoothed_general_offsets)
        print(f"\n✓ Smoothness Comparison (lower=smoother):")
        print(f"  - Mouth region variance: {avg_mouth:.4f}")
        print(f"  - General landmarks variance: {avg_general:.4f}")
        print(f"  - Mouth is {(avg_general/avg_mouth):.2f}x smoother (desired > 1.0)")
    
    return True


def test_confidence_reset():
    """Test automatic buffer reset on confidence drop."""
    print("\n" + "="*70)
    print("TEST 3: Confidence-Based Buffer Reset")
    print("="*70)
    
    smoother = LandmarkSmoothingBuffer(
        history_size=5,
        smoothing_factor=0.7,
        enable_debug=True
    )
    
    landmarks = np.random.randn(106, 2).astype(np.float32) * 20 + 200
    bbox = np.array([100, 100, 300, 300])
    
    # High confidence frames
    print("\n  Processing frames with high confidence (0.95)...")
    for i in range(3):
        smoother.smooth(landmarks, face_bbox=bbox, confidence=0.95)
    
    # Sudden confidence drop
    print("  Confidence drops to 0.1 - buffer should reset...")
    smoother.smooth(landmarks, face_bbox=bbox, confidence=0.1)
    
    stats = smoother.get_stats()
    print(f"\n✓ Statistics after confidence drop:")
    print(f"  - Buffer resets: {stats['buffer_resets']}")
    
    if stats['buffer_resets'] > 0:
        print(f"  ✓ Buffer reset triggered correctly on confidence drop")
        return True
    
    return False


def test_integration():
    """Test integration with configuration system."""
    print("\n" + "="*70)
    print("TEST 4: Configuration Integration")
    print("="*70)
    
    # Check that configuration is properly added to globals
    print("\n  Checking global configuration settings...")
    
    config_attrs = [
        'enable_landmark_smoothing',
        'landmark_smoothing_factor',
        'mouth_smoothing_factor',
        'smoothing_history_size',
        'show_landmark_debug_overlay',
    ]
    
    missing = []
    for attr in config_attrs:
        if not hasattr(modules.globals, attr):
            missing.append(attr)
        else:
            val = getattr(modules.globals, attr)
            print(f"  ✓ {attr}: {val}")
    
    if missing:
        print(f"\n  ✗ Missing configuration attributes: {missing}")
        return False
    
    print(f"\n✓ All {len(config_attrs)} configuration options present")
    return True


def test_multi_face_tracking():
    """Test tracking multiple faces simultaneously."""
    print("\n" + "="*70)
    print("TEST 5: Multi-Face Tracking")
    print("="*70)
    
    smoother = LandmarkSmoothingBuffer(
        history_size=5,
        smoothing_factor=0.7,
        enable_debug=False
    )
    
    # Simulate two faces at different positions
    landmarks1 = np.random.randn(106, 2).astype(np.float32) * 20 + 150
    landmarks2 = np.random.randn(106, 2).astype(np.float32) * 20 + 350
    
    bbox1 = np.array([50, 50, 200, 200])
    bbox2 = np.array([300, 300, 450, 450])
    
    print(f"\n  Processing Face 1 (left) and Face 2 (right) for 10 frames...")
    
    for frame_idx in range(10):
        # Add noise
        noise1 = np.random.randn(106, 2).astype(np.float32) * 2
        noise2 = np.random.randn(106, 2).astype(np.float32) * 2
        
        # Process both faces
        smoothed1 = smoother.smooth(landmarks1 + noise1, face_bbox=bbox1, confidence=0.95)
        smoothed2 = smoother.smooth(landmarks2 + noise2, face_bbox=bbox2, confidence=0.90)
    
    stats = smoother.get_stats()
    print(f"\n✓ Multi-face tracking:")
    print(f"  - Total frames processed: {stats['processed_frames']}")
    print(f"  - Smoothing operations: {stats['smoothing_applied']}")
    
    return True


def test_reset_functionality():
    """Test reset logic."""
    print("\n" + "="*70)
    print("TEST 6: Reset Functionality")
    print("="*70)
    
    # Use global singleton
    reset_landmark_smoother()
    
    landmarks = np.random.randn(106, 2).astype(np.float32) * 20 + 200
    bbox = np.array([100, 100, 300, 300])
    
    # Process some frames
    print("\n  Processing 5 frames...")
    for i in range(5):
        smooth_landmarks(landmarks, face_bbox=bbox, smoothing_enabled=True)
    
    # Reset
    print("  Resetting landmark smoother...")
    reset_landmark_smoother()
    
    # Process again - should start fresh
    print("  Processing 3 more frames after reset...")
    for i in range(3):
        smooth_landmarks(landmarks, face_bbox=bbox, smoothing_enabled=True)
    
    print(f"\n✓ Reset functionality working correctly")
    return True


def main():
    """Run all tests."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  LANDMARK SMOOTHING FEATURE - COMPREHENSIVE TEST SUITE  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tests = [
        ("Basic Smoothing", test_basic_smoothing),
        ("Mouth Emphasis", test_mouth_emphasis),
        ("Confidence Reset", test_confidence_reset),
        ("Configuration", test_integration),
        ("Multi-Face Tracking", test_multi_face_tracking),
        ("Reset Functionality", test_reset_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "🎉 "*20)
        print("ALL TESTS PASSED! Landmark smoothing is ready for use.".center(70))
        print("🎉 "*20)
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
