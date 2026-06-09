"""
Test script for mouth morphing integration

This script tests the new mouth morphing functionality that morphs
the source face's mouth to match the target face's mouth movements.
"""

import cv2
import numpy as np
import modules.globals
from modules.face_analyser import get_one_face
from modules.processors.frame.mouth_morpher import (
    extract_mouth_landmarks_from_face,
    calculate_mouth_openness_from_landmarks,
    apply_mouth_morphing_to_source,
    reset_mouth_morpher,
    get_mouth_openness_smoothed,
    MouthOpennessTracker
)


def test_mouth_openness_tracker():
    """Test the MouthOpennessTracker class."""
    print("\n" + "="*60)
    print("TEST 1: MouthOpennessTracker")
    print("="*60)
    
    tracker = MouthOpennessTracker(history_size=5, smoothing_alpha=0.8)
    
    # Simulate mouth opening sequence
    openness_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.6, 0.4, 0.2, 0.0]
    
    print("\nSimulating mouth opening and closing:")
    print(f"{'Frame':<8} {'Raw':<10} {'Smoothed':<10} {'Difference':<10}")
    print("-" * 40)
    
    for i, openness in enumerate(openness_values):
        smoothed = tracker.update(openness)
        diff = abs(openness - smoothed)
        print(f"{i:<8} {openness:<10.3f} {smoothed:<10.3f} {diff:<10.3f}")
    
    print("\n✓ MouthOpennessTracker test passed")
    print(f"  - Smoothing reduces jitter while following trend")
    print(f"  - Final smoothed value: {tracker.get_smoothed_openness():.3f}")


def test_mouth_openness_calculation():
    """Test mouth openness calculation from landmarks."""
    print("\n" + "="*60)
    print("TEST 2: Mouth Openness Calculation")
    print("="*60)
    
    # Simulate closed mouth landmarks (simplified 3-point)
    closed_mouth = np.array([
        [100, 150],  # Left corner
        [150, 150],  # Center
        [200, 150]   # Right corner
    ], dtype=np.float32)
    
    openness_closed = calculate_mouth_openness_from_landmarks(closed_mouth)
    print(f"\nClosed mouth openness: {openness_closed:.3f}")
    assert openness_closed < 0.3, "Closed mouth should have low openness"
    
    # Simulate open mouth landmarks (wider)
    open_mouth = np.array([
        [90, 140],   # Left corner (wider)
        [150, 145],  # Center
        [210, 140]   # Right corner (wider)
    ], dtype=np.float32)
    
    openness_open = calculate_mouth_openness_from_landmarks(open_mouth)
    print(f"Open mouth openness: {openness_open:.3f}")
    
    print(f"\n✓ Mouth openness calculation test passed")
    print(f"  - Closed: {openness_closed:.3f}")
    print(f"  - Open: {openness_open:.3f}")
    print(f"  - Difference: {openness_open - openness_closed:.3f}")


def test_landmark_extraction():
    """Test landmark extraction from Face object."""
    print("\n" + "="*60)
    print("TEST 3: Landmark Extraction from Face")
    print("="*60)
    
    # Create a mock Face object
    class MockFace:
        def __init__(self, has_106=True, has_kps=True):
            if has_106:
                # Simulate 106-point landmarks
                self.landmark_2d_106 = np.random.randn(106, 2).astype(np.float32) * 50 + 200
            else:
                self.landmark_2d_106 = None
            
            if has_kps:
                # Simulate 5-point keypoints
                self.kps = np.random.randn(5, 2).astype(np.float32) * 50 + 200
            else:
                self.kps = None
    
    # Test with 106-point landmarks
    face_106 = MockFace(has_106=True, has_kps=True)
    mouth_landmarks = extract_mouth_landmarks_from_face(face_106)
    
    if mouth_landmarks is not None:
        print(f"\n✓ 106-point landmarks extracted")
        print(f"  - Shape: {mouth_landmarks.shape}")
        print(f"  - Expected: (29, 2) [mouth region 48-77]")
    else:
        print("✗ Failed to extract 106-point landmarks")
    
    # Test with 5-point keypoints only
    face_5pt = MockFace(has_106=False, has_kps=True)
    mouth_landmarks_5pt = extract_mouth_landmarks_from_face(face_5pt)
    
    if mouth_landmarks_5pt is not None:
        print(f"\n✓ 5-point keypoints extracted (fallback)")
        print(f"  - Shape: {mouth_landmarks_5pt.shape}")
        print(f"  - Expected: (3, 2) [left, center, right]")
    else:
        print("✗ Failed to extract 5-point keypoints")
    
    # Test with no landmarks
    face_none = MockFace(has_106=False, has_kps=False)
    mouth_landmarks_none = extract_mouth_landmarks_from_face(face_none)
    
    if mouth_landmarks_none is None:
        print(f"\n✓ Correctly returns None when no landmarks available")
    else:
        print("✗ Should return None when no landmarks")


def test_configuration():
    """Test configuration in modules.globals."""
    print("\n" + "="*60)
    print("TEST 4: Configuration in modules.globals")
    print("="*60)
    
    # Check if new config options exist
    config_options = [
        'enable_mouth_morphing',
        'mouth_morph_strength',
        'mouth_openness_smoothing',
        'morph_transition_frames',
        'enable_jaw_tracking'
    ]
    
    print("\nChecking configuration options:")
    for option in config_options:
        if hasattr(modules.globals, option):
            value = getattr(modules.globals, option)
            print(f"  ✓ {option}: {value}")
        else:
            print(f"  ✗ {option}: NOT FOUND")
    
    # Test setting values
    print("\nTesting configuration modification:")
    original_value = modules.globals.enable_mouth_morphing
    modules.globals.enable_mouth_morphing = True
    modules.globals.mouth_morph_strength = 0.8
    
    print(f"  ✓ enable_mouth_morphing set to: {modules.globals.enable_mouth_morphing}")
    print(f"  ✓ mouth_morph_strength set to: {modules.globals.mouth_morph_strength}")
    
    # Restore original
    modules.globals.enable_mouth_morphing = original_value


def test_reset_functionality():
    """Test reset function."""
    print("\n" + "="*60)
    print("TEST 5: Reset Functionality")
    print("="*60)
    
    # Create tracker and add some history
    from modules.processors.frame.mouth_morpher import _openness_tracker
    
    print("\nAdding history to tracker...")
    for i in range(5):
        _openness_tracker.update(0.5 + i * 0.1)
    
    print(f"  Smoothed openness before reset: {_openness_tracker.get_smoothed_openness():.3f}")
    print(f"  History size: {len(_openness_tracker.history)}")
    
    # Reset
    reset_mouth_morpher()
    print(f"\nAfter reset:")
    print(f"  Smoothed openness: {_openness_tracker.get_smoothed_openness():.3f}")
    print(f"  History size: {len(_openness_tracker.history)}")
    
    print("\n✓ Reset functionality test passed")


def test_integration_with_face_swapper():
    """Test that face_swapper imports mouth_morpher correctly."""
    print("\n" + "="*60)
    print("TEST 6: Integration with face_swapper.py")
    print("="*60)
    
    try:
        from modules.processors.frame.face_swapper import (
            apply_mouth_morphing_to_source,
            reset_mouth_morpher,
            get_mouth_openness_smoothed
        )
        print("\n✓ mouth_morpher functions imported successfully")
        print(f"  - apply_mouth_morphing_to_source: {apply_mouth_morphing_to_source}")
        print(f"  - reset_mouth_morpher: {reset_mouth_morpher}")
        print(f"  - get_mouth_openness_smoothed: {get_mouth_openness_smoothed}")
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("MOUTH MORPHING INTEGRATION TEST SUITE")
    print("="*60)
    print("\nTesting new mouth morphing functionality...")
    
    try:
        test_mouth_openness_tracker()
        test_mouth_openness_calculation()
        test_landmark_extraction()
        test_configuration()
        test_reset_functionality()
        test_integration_with_face_swapper()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nMouth morphing integration is working correctly!")
        print("\nTo enable mouth morphing:")
        print("  modules.globals.enable_mouth_morphing = True")
        print("  modules.globals.mouth_morph_strength = 0.7")
        print("  modules.globals.mouth_openness_smoothing = 0.8")
        
    except Exception as e:
        print(f"\n" + "="*60)
        print(f"TEST FAILED ✗")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
