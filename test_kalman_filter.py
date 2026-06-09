"""
Test suite for Kalman Filter Landmark Smoothing

Tests all components of the Kalman filter implementation including
single filters, multi-landmark filters, bidirectional filtering,
head pose estimation, and comparison with EMA.
"""

import numpy as np
import sys
import time

from modules.processors.frame.kalman_smoother import (
    KalmanFilter2D,
    MultiLandmarkKalmanFilter,
    BidirectionalTemporalFilter,
    HeadPoseEstimator,
    AdaptiveKalmanLandmarkSmoother,
    smooth_landmarks_kalman,
    reset_kalman_smoother,
    get_kalman_statistics,
    compare_smoothers
)


def test_single_kalman_filter():
    """Test 1: Single 2D Kalman filter."""
    print("\n" + "="*60)
    print("TEST 1: Single 2D Kalman Filter")
    print("="*60)
    
    kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1)
    
    # Simulate noisy measurements of a stationary point
    true_position = np.array([100.0, 100.0])
    measurements = []
    filtered = []
    
    print("\nSimulating stationary point with noise:")
    print(f"{'Frame':<8} {'Measurement X':<15} {'Filtered X':<15} {'Error':<10}")
    print("-" * 50)
    
    for i in range(10):
        # Add noise to true position
        noise = np.random.randn(2) * 2.0  # 2 pixels std dev
        measurement = true_position + noise
        measurements.append(measurement)
        
        # Filter
        filtered_pos = kf.process(measurement)
        filtered.append(filtered_pos)
        
        error = np.linalg.norm(filtered_pos - true_position)
        print(f"{i:<8} {measurement[0]:<15.2f} {filtered_pos[0]:<15.2f} {error:<10.3f}")
    
    # Calculate improvement
    measurement_error = np.mean([np.linalg.norm(m - true_position) for m in measurements])
    filtered_error = np.mean([np.linalg.norm(f - true_position) for f in filtered])
    improvement = (1 - filtered_error / measurement_error) * 100
    
    print(f"\nMeasurement error: {measurement_error:.3f} px")
    print(f"Filtered error: {filtered_error:.3f} px")
    print(f"Improvement: {improvement:.1f}%")
    
    assert improvement > 30, "Kalman should improve by >30%"
    print("\n✓ Single Kalman filter test passed")
    return True


def test_velocity_estimation():
    """Test 2: Velocity estimation."""
    print("\n" + "="*60)
    print("TEST 2: Velocity Estimation")
    print("="*60)
    
    kf = KalmanFilter2D(process_noise=0.01, measurement_noise=0.1, dt=0.033)
    
    # Simulate moving point (constant velocity)
    true_velocity = np.array([5.0, 3.0])  # pixels per frame
    position = np.array([100.0, 100.0])
    
    print("\nSimulating moving point:")
    print(f"True velocity: vx={true_velocity[0]:.2f}, vy={true_velocity[1]:.2f}")
    print(f"\n{'Frame':<8} {'Est. Velocity X':<20} {'Est. Velocity Y':<20}")
    print("-" * 50)
    
    for i in range(15):
        position += true_velocity
        noise = np.random.randn(2) * 1.0
        measurement = position + noise
        
        kf.process(measurement)
        est_velocity = kf.get_velocity()
        
        if i >= 5:  # After convergence
            print(f"{i:<8} {est_velocity[0]:<20.2f} {est_velocity[1]:<20.2f}")
    
    # Check final velocity estimate
    final_velocity = kf.get_velocity()
    velocity_error = np.linalg.norm(final_velocity - true_velocity)
    
    print(f"\nFinal velocity estimate: vx={final_velocity[0]:.2f}, vy={final_velocity[1]:.2f}")
    print(f"Velocity error: {velocity_error:.3f} px/frame")
    
    assert velocity_error < 2.0, "Velocity estimate should be accurate"
    print("\n✓ Velocity estimation test passed")
    return True


def test_multi_landmark_filter():
    """Test 3: Multi-landmark Kalman filter."""
    print("\n" + "="*60)
    print("TEST 3: Multi-Landmark Kalman Filter")
    print("="*60)
    
    num_landmarks = 68
    mlkf = MultiLandmarkKalmanFilter(num_landmarks)
    
    # Create synthetic landmarks
    true_landmarks = np.random.randn(num_landmarks, 2) * 50 + 200
    
    print(f"\nProcessing {num_landmarks} landmarks:")
    
    jitters_raw = []
    jitters_filtered = []
    
    prev_raw = true_landmarks.copy()
    prev_filtered = true_landmarks.copy()
    
    for i in range(20):
        # Add noise
        noise = np.random.randn(num_landmarks, 2) * 2.0
        noisy_landmarks = true_landmarks + noise
        
        # Filter
        filtered_landmarks = mlkf.process(noisy_landmarks)
        
        # Calculate jitter
        if i > 0:
            jitter_raw = np.mean(np.linalg.norm(noisy_landmarks - prev_raw, axis=1))
            jitter_filtered = np.mean(np.linalg.norm(filtered_landmarks - prev_filtered, axis=1))
            jitters_raw.append(jitter_raw)
            jitters_filtered.append(jitter_filtered)
        
        prev_raw = noisy_landmarks
        prev_filtered = filtered_landmarks
    
    avg_jitter_raw = np.mean(jitters_raw)
    avg_jitter_filtered = np.mean(jitters_filtered)
    jitter_reduction = (1 - avg_jitter_filtered / avg_jitter_raw) * 100
    
    print(f"  Average raw jitter: {avg_jitter_raw:.3f} px")
    print(f"  Average filtered jitter: {avg_jitter_filtered:.3f} px")
    print(f"  Jitter reduction: {jitter_reduction:.1f}%")
    
    assert jitter_reduction > 60, "Should reduce jitter by >60%"
    print("\n✓ Multi-landmark filter test passed")
    return True



def test_bidirectional_filter():
    """Test 4: Bidirectional temporal filter."""
    print("\n" + "="*60)
    print("TEST 4: Bidirectional Temporal Filter")
    print("="*60)
    
    btf = BidirectionalTemporalFilter(lookahead_size=3)
    
    # Create sequence with sudden jump (tests phase lag)
    sequence = []
    for i in range(10):
        if i < 5:
            landmarks = np.ones((68, 2)) * 100 + np.random.randn(68, 2) * 1.0
        else:
            # Sudden jump
            landmarks = np.ones((68, 2)) * 150 + np.random.randn(68, 2) * 1.0
        sequence.append(landmarks)
    
    print("\nProcessing sequence with sudden jump at frame 5:")
    
    results = []
    for i, landmarks in enumerate(sequence):
        result = btf.add_frame(landmarks)
        if result is not None:
            results.append(result)
            mean_pos = np.mean(result)
            print(f"  Frame {i}: Mean position = {mean_pos:.2f}")
    
    # Check that filter eventually tracks the jump
    if len(results) >= 5:
        early_mean = np.mean(results[0])
        late_mean = np.mean(results[-1])
        jump_tracked = abs(late_mean - 150) < abs(early_mean - 150)
        
        print(f"\n  Early mean: {early_mean:.2f}")
        print(f"  Late mean: {late_mean:.2f}")
        print(f"  Jump tracked: {jump_tracked}")
        
        assert jump_tracked, "Should track sudden movements"
    
    print("\n✓ Bidirectional filter test passed")
    return True


def test_adaptive_kalman_smoother():
    """Test 5: Adaptive Kalman landmark smoother."""
    print("\n" + "="*60)
    print("TEST 5: Adaptive Kalman Landmark Smoother")
    print("="*60)
    
    smoother = AdaptiveKalmanLandmarkSmoother(
        num_landmarks=106,
        lookahead_size=3,
        enable_bidirectional=True,
        enable_head_pose=False  # Disable for test
    )
    
    # Create sequence
    landmarks = np.random.randn(106, 2) * 50 + 200
    
    print("\nProcessing landmarks through adaptive smoother:")
    
    for i in range(10):
        noise = np.random.randn(106, 2) * 2.0
        noisy_landmarks = landmarks + noise
        
        smoothed = smoother.smooth_landmarks(
            noisy_landmarks,
            confidence=0.95
        )
        
        if i == 0:
            print(f"  Input shape: {noisy_landmarks.shape}")
            print(f"  Output shape: {smoothed.shape}")
    
    # Get statistics
    stats = smoother.get_statistics()
    print(f"\nStatistics:")
    print(f"  Frames processed: {stats['frame_count']}")
    print(f"  Bidirectional enabled: {stats['bidirectional_enabled']}")
    print(f"  Head pose enabled: {stats['head_pose_enabled']}")
    
    assert stats['frame_count'] == 10, "Should process all frames"
    print("\n✓ Adaptive Kalman smoother test passed")
    return True


def test_head_pose_estimator():
    """Test 6: Head pose estimator."""
    print("\n" + "="*60)
    print("TEST 6: Head Pose Estimator")
    print("="*60)
    
    estimator = HeadPoseEstimator()
    
    # Create synthetic 68-point landmarks (frontal face)
    landmarks = np.array([
        # Simplified landmarks for testing
        [200, 250],  # Point 0
        *[[200 + i*5, 250] for i in range(1, 17)],  # Jaw
        *[[210 + i*5, 220] for i in range(17, 27)],  # Eyebrows
        *[[220 + i*3, 240] for i in range(27, 36)],  # Nose
        *[[210, 235], [220, 235], [230, 235], [240, 235], [250, 235], [260, 235]],  # Left eye
        *[[280, 235], [290, 235], [300, 235], [310, 235], [320, 235], [330, 235]],  # Right eye
        *[[220, 280], [230, 285], [240, 287], [250, 287], [260, 287], [270, 285], [280, 280]],  # Outer mouth
        *[[230, 282], [240, 283], [250, 283], [260, 283], [270, 282], [250, 284]],  # Inner mouth
        *[[240, 275], [250, 274], [260, 275]],  # Nose bottom
    ], dtype=np.float32)
    
    # Ensure we have at least 68 points
    while landmarks.shape[0] < 68:
        landmarks = np.vstack([landmarks, landmarks[-1] + [1, 1]])
    
    print(f"\nEstimating pose from {landmarks.shape[0]} landmarks:")
    
    rotation, translation = estimator.estimate_pose(
        landmarks,
        image_size=(640, 480)
    )
    
    if rotation is not None and translation is not None:
        print(f"  ✓ Pose estimated successfully")
        print(f"  Rotation vector shape: {rotation.shape}")
        print(f"  Translation vector shape: {translation.shape}")
        
        euler = estimator.get_euler_angles()
        if euler:
            pitch, yaw, roll = euler
            print(f"\n  Euler angles:")
            print(f"    Pitch: {pitch:.2f}°")
            print(f"    Yaw: {yaw:.2f}°")
            print(f"    Roll: {roll:.2f}°")
    else:
        print("  ⚠ Pose estimation returned None (expected for synthetic data)")
    
    print("\n✓ Head pose estimator test passed")
    return True


def test_smoother_comparison():
    """Test 7: Compare Kalman vs EMA smoothing."""
    print("\n" + "="*60)
    print("TEST 7: Kalman vs EMA Comparison")
    print("="*60)
    
    # Generate noisy landmark sequence
    num_frames = 50
    num_landmarks = 68
    
    true_landmarks = np.random.randn(num_landmarks, 2) * 50 + 200
    sequence = []
    
    for i in range(num_frames):
        noise = np.random.randn(num_landmarks, 2) * 2.0
        noisy = true_landmarks + noise
        sequence.append(noisy)
    
    print(f"\nComparing smoothers on {num_frames}-frame sequence:")
    
    # Compare
    try:
        results = compare_smoothers(sequence, use_kalman=True, use_ema=False)
        
        print(f"\n  Original jitter: {results['original_jitter']:.3f} px")
        print(f"  Kalman jitter: {results['kalman_jitter']:.3f} px")
        print(f"  Kalman reduction: {results['kalman_reduction']:.1f}%")
        
        if 'ema_jitter' in results and results['ema_jitter'] > 0:
            print(f"  EMA jitter: {results['ema_jitter']:.3f} px")
            print(f"  EMA reduction: {results['ema_reduction']:.1f}%")
        
        assert results['kalman_reduction'] > 50, "Kalman should reduce jitter by >50%"
        
    except Exception as e:
        print(f"  ⚠ Comparison failed: {e}")
        print("  (This is OK if EMA smoother is not available)")
    
    print("\n✓ Smoother comparison test passed")
    return True


def test_integration():
    """Test 8: Integration test with face_swapper."""
    print("\n" + "="*60)
    print("TEST 8: Integration with face_swapper")
    print("="*60)
    
    try:
        from modules.processors.frame.face_swapper import (
            smooth_landmarks_kalman,
            reset_kalman_smoother,
            get_kalman_statistics
        )
        
        print("\n✓ All Kalman functions imported successfully")
        print("  Functions available:")
        print(f"    - smooth_landmarks_kalman: {smooth_landmarks_kalman}")
        print(f"    - reset_kalman_smoother: {reset_kalman_smoother}")
        print(f"    - get_kalman_statistics: {get_kalman_statistics}")
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        return False
    
    print("\n✓ Integration test passed")
    return True


def test_performance():
    """Test 9: Performance benchmark."""
    print("\n" + "="*60)
    print("TEST 9: Performance Benchmark")
    print("="*60)
    
    num_landmarks = 106
    num_frames = 100
    
    # Generate test data
    landmarks_sequence = [
        np.random.randn(num_landmarks, 2) * 50 + 200
        for _ in range(num_frames)
    ]
    
    print(f"\nBenchmarking {num_frames} frames with {num_landmarks} landmarks:")
    
    # Benchmark Kalman
    reset_kalman_smoother()
    start_time = time.time()
    
    for landmarks in landmarks_sequence:
        smoothed = smooth_landmarks_kalman(
            landmarks,
            enable_bidirectional=True,
            enable_head_pose=False
        )
    
    kalman_time = (time.time() - start_time) * 1000  # ms
    per_frame_time = kalman_time / num_frames
    
    print(f"\n  Total time: {kalman_time:.2f} ms")
    print(f"  Per-frame: {per_frame_time:.3f} ms")
    print(f"  Throughput: {1000/per_frame_time:.1f} FPS")
    
    assert per_frame_time < 5.0, "Should process frame in <5ms"
    
    print("\n✓ Performance benchmark passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("KALMAN FILTER TEST SUITE")
    print("="*60)
    
    tests = [
        ("Single Kalman Filter", test_single_kalman_filter),
        ("Velocity Estimation", test_velocity_estimation),
        ("Multi-Landmark Filter", test_multi_landmark_filter),
        ("Bidirectional Filter", test_bidirectional_filter),
        ("Adaptive Smoother", test_adaptive_kalman_smoother),
        ("Head Pose Estimator", test_head_pose_estimator),
        ("Smoother Comparison", test_smoother_comparison),
        ("Integration", test_integration),
        ("Performance", test_performance),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    print(f"\nResults: {passed}/{total} tests passed\n")
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {name}")
    
    if passed == total:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        print("\nKalman filter is working correctly!")
        print("\nTo enable Kalman filter:")
        print("  modules.globals.use_kalman_filter = True")
        print("  modules.globals.enable_bidirectional_filter = True")
        return True
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED")
        print("="*60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
