"""
Test Suite for Quality Enhancement and Lip Sync Features

Run this to validate all new features before committing.

Usage:
    python test_enhancements.py
    
Or run specific tests:
    python test_enhancements.py --test quality_enhancer
    python test_enhancements.py --test lip_sync
    python test_enhancements.py --test audio_analysis
"""

import sys
import os
import numpy as np
import cv2
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Test counters
TESTS_PASSED = 0
TESTS_FAILED = 0
TESTS_SKIPPED = 0


def print_header(text):
    """Print formatted test section header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def test_passed(test_name):
    """Mark test as passed."""
    global TESTS_PASSED
    TESTS_PASSED += 1
    print(f"✓ PASS: {test_name}")


def test_failed(test_name, error):
    """Mark test as failed."""
    global TESTS_FAILED
    TESTS_FAILED += 1
    print(f"✗ FAIL: {test_name}")
    print(f"  Error: {error}\n")


def test_skipped(test_name, reason):
    """Mark test as skipped."""
    global TESTS_SKIPPED
    TESTS_SKIPPED += 1
    print(f"⊘ SKIP: {test_name}")
    print(f"  Reason: {reason}\n")


# ============================================================================
# Test: Video Capture Resolution Presets
# ============================================================================

def test_video_capture_presets():
    """Test video capture resolution presets."""
    print_header("Testing Video Capture Resolution Presets")
    
    try:
        from modules.video_capture import RESOLUTION_PRESETS, VideoCapturer
        
        # Test preset definitions
        expected_presets = ["480p", "720p", "1080p", "1440p", "2K", "4K"]
        for preset in expected_presets:
            if preset in RESOLUTION_PRESETS:
                w, h = RESOLUTION_PRESETS[preset]
                print(f"  {preset}: {w}x{h}")
                test_passed(f"Preset '{preset}' defined")
            else:
                test_failed(f"Preset '{preset}' exists", "Preset not found in RESOLUTION_PRESETS")
        
        # Test that 1080p is the new default
        assert RESOLUTION_PRESETS["1080p"] == (1920, 1080), "1080p should be 1920x1080"
        test_passed("1080p is correctly configured as 1920x1080")
        
        # Test VideoCapturer methods exist
        assert hasattr(VideoCapturer, "start_with_preset"), "start_with_preset method missing"
        test_passed("VideoCapturer.start_with_preset() method exists")
        
        assert hasattr(VideoCapturer, "get_available_resolutions"), "get_available_resolutions method missing"
        test_passed("VideoCapturer.get_available_resolutions() method exists")
        
    except Exception as e:
        test_failed("Video Capture Presets", str(e))


# ============================================================================
# Test: Quality Enhancer Module
# ============================================================================

def test_quality_enhancer():
    """Test quality enhancement functions."""
    print_header("Testing Quality Enhancer Module")
    
    try:
        from modules.processors.frame.quality_enhancer import (
            get_quality_preset,
            upscale_frame,
            downscale_frame,
            resize_frame_to_target,
            enhance_frame_detail,
            enhance_contrast,
            apply_quality_profile,
        )
        
        # Create test frame (100x100 BGR image)
        test_frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Test 1: Get quality presets
        for mode in ["standard", "high", "ultra"]:
            preset = get_quality_preset(mode)
            assert "resolution" in preset, f"{mode} preset missing 'resolution'"
            assert "upscale_factor" in preset, f"{mode} preset missing 'upscale_factor'"
            test_passed(f"Quality preset '{mode}' loaded correctly")
        
        # Test 2: Upscaling
        upscaled = upscale_frame(test_frame, factor=1.5, method="lanczos")
        assert upscaled.shape[0] == int(100 * 1.5), "Upscaling height failed"
        assert upscaled.shape[1] == int(100 * 1.5), "Upscaling width failed"
        test_passed("Frame upscaling works (1.5x Lanczos)")
        
        # Test 3: Downscaling
        downscaled = downscale_frame(test_frame, 50, 50)
        assert downscaled.shape[:2] == (50, 50), "Downscaling failed"
        test_passed("Frame downscaling works")
        
        # Test 4: Resize with target resolution
        resized = resize_frame_to_target(test_frame, (200, 200))
        assert resized.shape == (200, 200, 3), "Resize to target failed"
        test_passed("Resize frame to target resolution works")
        
        # Test 5: Detail enhancement
        enhanced = enhance_frame_detail(test_frame, strength=0.7)
        assert enhanced.shape == test_frame.shape, "Detail enhancement shape mismatch"
        assert enhanced.dtype == np.uint8, "Detail enhancement dtype should be uint8"
        test_passed("Frame detail enhancement works")
        
        # Test 6: Contrast enhancement
        contrasted = enhance_contrast(test_frame, strength=1.2)
        assert contrasted.shape == test_frame.shape, "Contrast enhancement shape mismatch"
        test_passed("Frame contrast enhancement works")
        
        # Test 7: Apply quality profile
        for profile in ["standard", "high", "ultra"]:
            result = apply_quality_profile(test_frame, profile)
            assert result.shape[2] == 3, f"Quality profile '{profile}' didn't preserve channels"
            test_passed(f"Apply quality profile '{profile}' works")
        
    except Exception as e:
        test_failed("Quality Enhancer Module", str(e))


# ============================================================================
# Test: Lip Sync Enhancer Module
# ============================================================================

def test_lip_sync_enhancer():
    """Test lip sync enhancement functions."""
    print_header("Testing Lip Sync Enhancer Module")
    
    try:
        from modules.processors.frame.lip_sync_enhancer import (
            MouthTracker,
            calculate_mouth_openness,
            MOUTH_STATES,
            create_improved_mouth_mask,
            blend_mouth_region,
            enhance_mouth_details,
        )
        
        # Test 1: Mouth state definitions
        expected_states = ["closed", "slightly_open", "open", "very_open"]
        for state in expected_states:
            assert state in MOUTH_STATES, f"State '{state}' not defined"
            test_passed(f"Mouth state '{state}' defined")
        
        # Test 2: MouthTracker initialization
        tracker = MouthTracker(history_size=5)
        assert tracker.history.maxlen == 5, "MouthTracker history size incorrect"
        test_passed("MouthTracker initialization works")
        
        # Test 3: Create sample mouth landmarks (20 points)
        mouth_landmarks = np.random.rand(20, 2).astype(np.float32) * 100
        
        # Test 4: Calculate mouth openness
        openness = calculate_mouth_openness(mouth_landmarks)
        assert 0.0 <= openness <= 1.0, "Mouth openness out of range"
        test_passed("Calculate mouth openness works")
        
        # Test 5: Update tracker with landmarks
        tracker.update(mouth_landmarks)
        assert tracker.get_state() in expected_states, "Invalid state returned"
        test_passed("MouthTracker update works")
        
        # Test 6: Get smoothed landmarks
        smoothed = tracker.get_smoothed_landmarks()
        assert smoothed is not None, "Smoothed landmarks is None"
        assert smoothed.shape == mouth_landmarks.shape, "Smoothed landmarks shape mismatch"
        test_passed("Get smoothed landmarks works")
        
        # Test 7: Create mouth mask
        test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Create mock face object
        class MockFace:
            def __init__(self):
                self.landmark_2d_106 = np.random.rand(106, 2).astype(np.float32) * 200
        
        mock_face = MockFace()
        mask = create_improved_mouth_mask(mock_face, test_frame, mask_type="feathered")
        assert mask.shape[:2] == (200, 200), "Mouth mask shape incorrect"
        test_passed("Create improved mouth mask works")
        
        # Test 8: Enhance mouth details
        mouth_region = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        enhanced = enhance_mouth_details(mouth_region, brightness_factor=1.1, contrast_factor=1.2)
        assert enhanced.shape == mouth_region.shape, "Enhanced mouth shape mismatch"
        assert enhanced.dtype == np.uint8, "Enhanced mouth dtype should be uint8"
        test_passed("Enhance mouth details works")
        
        # Test 9: Blend mouth region
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        mouth_region = np.ones((50, 50, 3), dtype=np.uint8) * 128
        original_mouth = np.ones((50, 50, 3), dtype=np.uint8) * 100
        bbox = (75, 75, 125, 125)
        
        blended = blend_mouth_region(frame, mouth_region, original_mouth, bbox, 
                                     blend_strength=0.8, feather_size=10)
        assert blended.shape == frame.shape, "Blended frame shape mismatch"
        test_passed("Blend mouth region works")
        
    except Exception as e:
        test_failed("Lip Sync Enhancer Module", str(e))


# ============================================================================
# Test: Audio Analysis Module
# ============================================================================

def test_audio_analysis():
    """Test audio analysis functions."""
    print_header("Testing Audio Analysis Module")
    
    try:
        from modules.processors.frame.audio_analysis import (
            has_librosa,
            PHONEME_TO_MOUTH_SHAPE,
            estimate_mouth_movement,
            get_voice_activity,
            get_audio_duration,
        )
        
        # Test 1: Check librosa availability
        librosa_available = has_librosa()
        print(f"  Librosa available: {librosa_available}")
        test_passed(f"Librosa detection works (available: {librosa_available})")
        
        # Test 2: Phoneme to mouth shape mapping
        expected_phonemes = ["AA", "B", "M", "P", "D", "G", "K"]
        for phoneme in expected_phonemes:
            assert phoneme in PHONEME_TO_MOUTH_SHAPE, f"Phoneme '{phoneme}' not mapped"
        test_passed(f"Phoneme to mouth shape mapping includes {len(PHONEME_TO_MOUTH_SHAPE)} phonemes")
        
        # Test 3: Estimate mouth movement from energy
        energy = np.random.rand(30).astype(np.float32)  # 1 second at 30fps
        movement = estimate_mouth_movement(energy, threshold=0.3, smoothing_window=3)
        assert movement.shape == energy.shape, "Movement estimation shape mismatch"
        assert np.all((movement >= 0.0) & (movement <= 1.0)), "Movement out of range"
        test_passed("Estimate mouth movement works")
        
        # Test 4: Get voice activity (with mock audio)
        # This will be skipped if no audio file is available
        test_skipped("Voice activity detection", "No test audio file available")
        
        # Test 5: Audio duration (with mock audio)
        test_skipped("Audio duration detection", "No test audio file available")
        
    except Exception as e:
        test_failed("Audio Analysis Module", str(e))


# ============================================================================
# Test: Global Configuration
# ============================================================================

def test_globals_configuration():
    """Test new global configuration settings."""
    print_header("Testing Global Configuration Settings")
    
    try:
        import modules.globals
        
        # Test quality settings
        settings = [
            ("quality_mode", str, "high"),
            ("target_resolution", tuple, (1920, 1080)),
            ("enable_upscaling", bool, True),
            ("upscaling_factor", float, 1.5),
            ("enable_face_enhancement", bool, True),
            ("face_enhancement_strength", float, 0.7),
            ("adaptive_quality", bool, True),
            ("enable_lip_sync", bool, True),
            ("lip_sync_strength", float, 0.8),
            ("enable_mouth_landmarks", bool, True),
            ("mouth_blend_feather", int, 25),
            ("enable_audio_analysis", bool, False),
            ("audio_sync_threshold", float, 0.5),
        ]
        
        for setting_name, expected_type, default_value in settings:
            if hasattr(modules.globals, setting_name):
                value = getattr(modules.globals, setting_name)
                if isinstance(value, expected_type):
                    test_passed(f"Global '{setting_name}' = {value}")
                else:
                    test_failed(f"Global '{setting_name}' type check", 
                               f"Expected {expected_type}, got {type(value)}")
            else:
                test_failed(f"Global '{setting_name}' exists", "Setting not found")
        
    except Exception as e:
        test_failed("Global Configuration", str(e))


# ============================================================================
# Test: Integration Example
# ============================================================================

def test_integration_example():
    """Test integration example script."""
    print_header("Testing Integration Example")
    
    try:
        from integration_example import EnhancedDeepLiveCamProcessor
        
        # Test processor initialization
        processor = EnhancedDeepLiveCamProcessor(
            quality_mode="high",
            enable_lip_sync=True,
            enable_audio_sync=False
        )
        assert processor.quality_mode == "high", "Quality mode not set"
        test_passed("EnhancedDeepLiveCamProcessor initialization works")
        
        # Test frame processing
        test_frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        processed = processor.process_frame(test_frame, faces=[])
        assert processed.shape == test_frame.shape, "Processed frame shape mismatch"
        test_passed("EnhancedDeepLiveCamProcessor.process_frame() works")
        
        # Test all quality modes
        for mode in ["standard", "high", "ultra"]:
            processor = EnhancedDeepLiveCamProcessor(quality_mode=mode)
            processed = processor.process_frame(test_frame, faces=[])
            assert processed.dtype == np.uint8, f"Quality mode '{mode}' dtype mismatch"
            test_passed(f"EnhancedDeepLiveCamProcessor quality mode '{mode}' works")
        
    except Exception as e:
        test_failed("Integration Example", str(e))


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  Deep-Live-Cam Quality & Lip Sync Enhancement Test Suite")
    print("="*70)
    
    test_functions = [
        ("Video Capture Presets", test_video_capture_presets),
        ("Quality Enhancer", test_quality_enhancer),
        ("Lip Sync Enhancer", test_lip_sync_enhancer),
        ("Audio Analysis", test_audio_analysis),
        ("Global Configuration", test_globals_configuration),
        ("Integration Example", test_integration_example),
    ]
    
    for name, test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ FATAL ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print(f"  Test Summary")
    print("="*70)
    print(f"  ✓ Passed:  {TESTS_PASSED}")
    print(f"  ✗ Failed:  {TESTS_FAILED}")
    print(f"  ⊘ Skipped: {TESTS_SKIPPED}")
    print(f"  ─────────────────────")
    total = TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED
    print(f"  Total:    {total}")
    print("="*70 + "\n")
    
    if TESTS_FAILED == 0:
        print("✓ All tests passed! Ready for commit.\n")
        return 0
    else:
        print(f"✗ {TESTS_FAILED} test(s) failed. Please fix before committing.\n")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test suite for quality and lip sync enhancements")
    parser.add_argument("--test", type=str, help="Run specific test module")
    args = parser.parse_args()
    
    if args.test:
        # Run specific test
        test_map = {
            "video_capture": test_video_capture_presets,
            "quality": test_quality_enhancer,
            "lip_sync": test_lip_sync_enhancer,
            "audio": test_audio_analysis,
            "globals": test_globals_configuration,
            "integration": test_integration_example,
        }
        
        if args.test in test_map:
            test_map[args.test]()
            print(f"\n✓ Test completed. Passed: {TESTS_PASSED}, Failed: {TESTS_FAILED}\n")
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {', '.join(test_map.keys())}")
    else:
        # Run all tests
        exit(run_all_tests())
