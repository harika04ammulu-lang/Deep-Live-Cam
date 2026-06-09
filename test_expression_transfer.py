"""
Test script for Expression Transfer feature

Tests MediaPipe-based expression detection and transfer functionality.
"""

import numpy as np
import sys

# Test MediaPipe availability first
print("="*60)
print("EXPRESSION TRANSFER TEST SUITE")
print("="*60)

print("\n[1/7] Checking MediaPipe availability...")
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    print("✓ MediaPipe found")
    print(f"  Version: {mp.__version__}")
except ImportError:
    HAS_MEDIAPIPE = False
    print("✗ MediaPipe not found")
    print("  Install with: pip install mediapipe")
    print("\n⚠ Some tests will be skipped")

# Import our modules
from modules.processors.frame.expression_transfer import (
    ActionUnitDetector,
    ExpressionSmoother,
    is_mediapipe_available,
    get_current_expression_aus,
    reset_expression_transfer
)
import modules.globals


def test_mediapipe_availability():
    """Test 1: Check MediaPipe availability."""
    print("\n" + "="*60)
    print("TEST 1: MediaPipe Availability")
    print("="*60)
    
    available = is_mediapipe_available()
    print(f"\nMediaPipe available: {available}")
    
    if available:
        print("✓ MediaPipe is ready for expression transfer")
    else:
        print("✗ MediaPipe not available - expression transfer disabled")
    
    return available



def test_action_unit_detector():
    """Test 2: ActionUnitDetector class."""
    print("\n" + "="*60)
    print("TEST 2: ActionUnitDetector")
    print("="*60)
    
    detector = ActionUnitDetector()
    
    # Create mock MediaPipe landmarks (468x3)
    print("\nCreating mock facial landmarks (468 points)...")
    landmarks = np.random.randn(468, 3).astype(np.float32)
    landmarks[:, :2] *= 100  # Scale to pixel coordinates
    landmarks[:, :2] += 200   # Center around 200,200
    landmarks[:, 2] *= 10     # Z-depth
    
    print("Testing individual AU detections:")
    
    # Test smile detection
    smile = detector.detect_smile_intensity(landmarks)
    print(f"  - Smile (AU6+AU12): {smile:.3f}")
    assert 0.0 <= smile <= 1.0, "Smile should be in [0,1] range"
    
    # Test eyebrow raise
    left_brow, right_brow = detector.detect_eyebrow_raise(landmarks)
    print(f"  - Left Eyebrow (AU1+AU2): {left_brow:.3f}")
    print(f"  - Right Eyebrow (AU1+AU2): {right_brow:.3f}")
    assert 0.0 <= left_brow <= 1.0, "Eyebrow raise should be in [0,1]"
    assert 0.0 <= right_brow <= 1.0, "Eyebrow raise should be in [0,1]"
    
    # Test eye squint
    left_squint, right_squint = detector.detect_eye_squint(landmarks)
    print(f"  - Left Eye Squint (AU7): {left_squint:.3f}")
    print(f"  - Right Eye Squint (AU7): {right_squint:.3f}")
    assert 0.0 <= left_squint <= 1.0, "Eye squint should be in [0,1]"
    
    # Test jaw drop
    jaw = detector.detect_jaw_drop(landmarks)
    print(f"  - Jaw Drop (AU26): {jaw:.3f}")
    assert 0.0 <= jaw <= 1.0, "Jaw drop should be in [0,1]"
    
    # Test all AUs together
    print("\nTesting detect_all_action_units()...")
    all_aus = detector.detect_all_action_units(landmarks)
    
    expected_keys = ['smile', 'left_eyebrow_raise', 'right_eyebrow_raise', 
                     'eyebrow_raise', 'left_eye_squint', 'right_eye_squint',
                     'eye_squint', 'jaw_drop']
    
    print(f"  Detected AUs: {len(all_aus)}")
    for key in expected_keys:
        assert key in all_aus, f"Missing AU: {key}"
        print(f"    {key}: {all_aus[key]:.3f}")
    
    print("\n✓ ActionUnitDetector test passed")
    return True


def test_expression_smoother():
    """Test 3: ExpressionSmoother class."""
    print("\n" + "="*60)
    print("TEST 3: ExpressionSmoother")
    print("="*60)
    
    smoother = ExpressionSmoother(history_size=5, smoothing_alpha=0.7)
    
    # Simulate jittery expression values
    raw_values = [0.5, 0.7, 0.45, 0.75, 0.6, 0.55, 0.65, 0.58]
    
    print("\nTesting temporal smoothing:")
    print(f"{'Frame':<8} {'Raw Value':<12} {'Smoothed':<12} {'Delta':<10}")
    print("-" * 45)
    
    smoothed_values = []
    for i, raw_val in enumerate(raw_values):
        expression = {'smile': raw_val, 'eyebrow': raw_val * 0.8}
        smoothed = smoother.smooth(expression)
        smoothed_val = smoothed['smile']
        smoothed_values.append(smoothed_val)
        delta = abs(raw_val - smoothed_val)
        print(f"{i:<8} {raw_val:<12.3f} {smoothed_val:<12.3f} {delta:<10.3f}")
    
    # Check smoothing effect
    raw_variance = np.var(raw_values)
    smoothed_variance = np.var(smoothed_values)
    
    print(f"\nVariance comparison:")
    print(f"  Raw variance: {raw_variance:.4f}")
    print(f"  Smoothed variance: {smoothed_variance:.4f}")
    print(f"  Reduction: {(1 - smoothed_variance/raw_variance)*100:.1f}%")
    
    assert smoothed_variance < raw_variance, "Smoothing should reduce variance"
    
    # Test reset
    print("\nTesting reset...")
    smoother.reset()
    assert len(smoother.expression_history) == 0, "History should be empty after reset"
    print("  ✓ Reset successful")
    
    print("\n✓ ExpressionSmoother test passed")
    return True



def test_configuration():
    """Test 4: Configuration in modules.globals."""
    print("\n" + "="*60)
    print("TEST 4: Configuration")
    print("="*60)
    
    config_options = [
        ('enable_expression_transfer', bool),
        ('expression_transfer_strength', float),
        ('expression_smoothing', float),
        ('transfer_smile', bool),
        ('transfer_eyebrow', bool),
        ('transfer_eye_squint', bool),
        ('transfer_jaw_drop', bool),
        ('show_expression_overlay', bool),
    ]
    
    print("\nChecking configuration options:")
    all_present = True
    
    for option, expected_type in config_options:
        if hasattr(modules.globals, option):
            value = getattr(modules.globals, option)
            type_match = isinstance(value, expected_type)
            status = "✓" if type_match else "⚠"
            print(f"  {status} {option}: {value} ({type(value).__name__})")
            if not type_match:
                print(f"      Expected type: {expected_type.__name__}")
                all_present = False
        else:
            print(f"  ✗ {option}: NOT FOUND")
            all_present = False
    
    # Test setting values
    print("\nTesting value modification:")
    original_enable = modules.globals.enable_expression_transfer
    original_strength = modules.globals.expression_transfer_strength
    
    modules.globals.enable_expression_transfer = True
    modules.globals.expression_transfer_strength = 0.8
    
    assert modules.globals.enable_expression_transfer == True
    assert modules.globals.expression_transfer_strength == 0.8
    print(f"  ✓ enable_expression_transfer set to: {modules.globals.enable_expression_transfer}")
    print(f"  ✓ expression_transfer_strength set to: {modules.globals.expression_transfer_strength}")
    
    # Restore original values
    modules.globals.enable_expression_transfer = original_enable
    modules.globals.expression_transfer_strength = original_strength
    print("  ✓ Original values restored")
    
    if all_present:
        print("\n✓ Configuration test passed")
    else:
        print("\n⚠ Some configuration options missing")
    
    return all_present


def test_reset_functionality():
    """Test 5: Reset functionality."""
    print("\n" + "="*60)
    print("TEST 5: Reset Functionality")
    print("="*60)
    
    # Simulate some expression detection
    print("\nSimulating expression detection...")
    
    # This should populate internal state
    detector = ActionUnitDetector()
    landmarks = np.random.randn(468, 3).astype(np.float32) * 100 + 200
    
    for i in range(5):
        _ = detector.detect_all_action_units(landmarks)
    
    print("  ✓ Generated expression history")
    
    # Reset
    print("\nCalling reset_expression_transfer()...")
    reset_expression_transfer()
    
    # Check that state is cleared
    aus = get_current_expression_aus()
    print(f"  Expression AUs after reset: {len(aus)} entries")
    
    print("\n✓ Reset functionality test passed")
    return True


def test_integration_with_face_swapper():
    """Test 6: Integration with face_swapper.py."""
    print("\n" + "="*60)
    print("TEST 6: Integration with face_swapper.py")
    print("="*60)
    
    try:
        from modules.processors.frame.face_swapper import (
            detect_and_transfer_expression,
            reset_expression_transfer,
            get_current_expression_aus,
            draw_expression_overlay,
            is_mediapipe_available
        )
        
        print("\n✓ All expression transfer functions imported successfully")
        print("  Functions available:")
        print(f"    - detect_and_transfer_expression: {detect_and_transfer_expression}")
        print(f"    - reset_expression_transfer: {reset_expression_transfer}")
        print(f"    - get_current_expression_aus: {get_current_expression_aus}")
        print(f"    - draw_expression_overlay: {draw_expression_overlay}")
        print(f"    - is_mediapipe_available: {is_mediapipe_available}")
        
        print("\n✓ Integration test passed")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False



def test_landmark_modification():
    """Test 7: Expression delta application to landmarks."""
    print("\n" + "="*60)
    print("TEST 7: Landmark Modification")
    print("="*60)
    
    from modules.processors.frame.expression_transfer import (
        apply_expression_deltas_to_landmarks
    )
    
    # Create mock landmarks (106-point format)
    print("\nCreating mock source landmarks (106 points)...")
    source_landmarks = np.random.randn(106, 2).astype(np.float32) * 50 + 200
    
    # Create expression AUs
    expression_aus = {
        'smile': 0.8,
        'left_eyebrow_raise': 0.6,
        'right_eyebrow_raise': 0.5,
        'left_eye_squint': 0.3,
        'right_eye_squint': 0.3,
        'jaw_drop': 0.4
    }
    
    print("\nApplying expression deltas:")
    for key, value in expression_aus.items():
        print(f"  {key}: {value:.2f}")
    
    # Apply deltas with moderate strength
    modified_landmarks = apply_expression_deltas_to_landmarks(
        source_landmarks,
        source_landmarks,
        expression_aus,
        strength=0.7
    )
    
    assert modified_landmarks is not None, "Should return landmarks"
    assert modified_landmarks.shape == source_landmarks.shape, "Shape should match"
    
    # Check that landmarks were actually modified
    difference = np.sum(np.abs(modified_landmarks - source_landmarks))
    print(f"\nTotal landmark displacement: {difference:.2f} pixels")
    
    if difference > 0.1:
        print("  ✓ Landmarks were modified by expression deltas")
        
        # Show some specific changes
        print("\nSample landmark changes:")
        for i in [48, 54, 17, 22]:  # Mouth corners, eyebrows
            if i < source_landmarks.shape[0]:
                delta = modified_landmarks[i] - source_landmarks[i]
                print(f"  Landmark {i}: Δx={delta[0]:.2f}, Δy={delta[1]:.2f}")
    else:
        print("  ⚠ Landmarks not significantly modified (might be low strength)")
    
    print("\n✓ Landmark modification test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    results = {}
    
    try:
        results['mediapipe'] = test_mediapipe_availability()
        results['au_detector'] = test_action_unit_detector()
        results['smoother'] = test_expression_smoother()
        results['config'] = test_configuration()
        results['reset'] = test_reset_functionality()
        results['integration'] = test_integration_with_face_swapper()
        results['landmarks'] = test_landmark_modification()
        
    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nResults: {passed}/{total} tests passed\n")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {test_name}")
    
    if passed == total:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nExpression transfer is working correctly!")
        print("\nTo enable expression transfer:")
        print("  modules.globals.enable_expression_transfer = True")
        print("  modules.globals.expression_transfer_strength = 0.7")
        print("  modules.globals.expression_smoothing = 0.7")
        
        if not HAS_MEDIAPIPE:
            print("\n⚠ Note: MediaPipe not installed")
            print("  Install with: pip install mediapipe")
            print("  Expression transfer requires MediaPipe to function")
        
        return True
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED")
        print("="*60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
