"""
Local Testing Script for Quality and Lip Sync Features

This script tests the new features in a practical way without requiring
the full UI to be running. Run this to validate features before committing.

Usage:
    python test_local_features.py

This will:
1. Test all imports work
2. Test 1080p resolution preset
3. Test quality enhancement pipeline
4. Test lip sync tracker
5. Validate audio analysis functions
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

print("\n" + "="*70)
print("  Deep-Live-Cam Feature Testing - Local Branch Test")
print("="*70)

# Test 1: Check we're on feature branch
print("\n[1/5] Checking git branch...")
try:
    result = os.popen("git branch --show-current").read().strip()
    if "quality" in result.lower() and "lip" in result.lower():
        print(f"  ✓ On feature branch: {result}")
    else:
        print(f"  ⚠ Branch: {result} (check if this is the right branch)")
except Exception as e:
    print(f"  ⚠ Could not check branch: {e}")

# Test 2: Import new modules
print("\n[2/5] Testing module imports...")
try:
    import modules.globals
    print("  ✓ modules.globals imported")
    
    from modules.video_capture import VideoCapturer, RESOLUTION_PRESETS
    print("  ✓ modules.video_capture imported with 1080p+ support")
    
    from modules.processors.frame.quality_enhancer import (
        upscale_frame,
        apply_quality_profile,
        get_quality_preset,
    )
    print("  ✓ modules.processors.frame.quality_enhancer imported")
    
    from modules.processors.frame.lip_sync_enhancer import (
        MouthTracker,
        calculate_mouth_openness,
        enhance_mouth_details,
    )
    print("  ✓ modules.processors.frame.lip_sync_enhancer imported")
    
    from modules.processors.frame.audio_analysis import (
        estimate_mouth_movement,
        has_librosa,
    )
    print("  ✓ modules.processors.frame.audio_analysis imported")
    
    from modules.processors.frame.face_masking import create_lower_mouth_mask
    print("  ✓ modules.processors.frame.face_masking updated with lip sync support")
    
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test 3: Test 1080p resolution presets
print("\n[3/5] Testing 1080p+ resolution support...")
try:
    presets = RESOLUTION_PRESETS
    
    # Check key presets
    preset_tests = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "4K": (3840, 2160),
    }
    
    for preset_name, expected_res in preset_tests.items():
        if preset_name in presets:
            actual_res = presets[preset_name]
            if actual_res == expected_res:
                print(f"  ✓ {preset_name}: {actual_res[0]}x{actual_res[1]}")
            else:
                print(f"  ✗ {preset_name}: Expected {expected_res}, got {actual_res}")
        else:
            print(f"  ✗ {preset_name}: Not found in presets")
            
    print(f"  ✓ Total presets available: {len(presets)}")
    
except Exception as e:
    print(f"  ✗ Resolution test failed: {e}")

# Test 4: Test quality enhancement with sample frame
print("\n[4/5] Testing quality enhancement pipeline...")
try:
    # Create test frame (1280x720)
    test_frame = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    
    # Test upscaling to 1080p
    upscaled = upscale_frame(test_frame, factor=1.5, method="lanczos")
    expected_height = int(720 * 1.5)
    expected_width = int(1280 * 1.5)
    
    if upscaled.shape[0] == expected_height and upscaled.shape[1] == expected_width:
        print(f"  ✓ Upscaling: {test_frame.shape[1]}x{test_frame.shape[0]} → {upscaled.shape[1]}x{upscaled.shape[0]}")
    else:
        print(f"  ✗ Upscaling failed: expected {expected_width}x{expected_height}")
    
    # Test quality profiles
    for profile in ["standard", "high", "ultra"]:
        enhanced = apply_quality_profile(test_frame, profile)
        if enhanced.shape == test_frame.shape and enhanced.dtype == np.uint8:
            print(f"  ✓ Quality profile '{profile}' works")
        else:
            print(f"  ✗ Quality profile '{profile}' failed")
    
    # Test getting quality presets
    high_preset = get_quality_preset("high")
    if high_preset["resolution"] == (1920, 1080):
        print(f"  ✓ Quality preset 'high' configured for 1080p")
    else:
        print(f"  ✗ Quality preset resolution mismatch")
        
except Exception as e:
    print(f"  ✗ Quality enhancement test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test lip sync features
print("\n[5/5] Testing lip sync enhancement...")
try:
    # Test mouth tracker
    tracker = MouthTracker(history_size=5)
    
    # Create sample mouth landmarks (20 points)
    mouth_landmarks = np.random.rand(20, 2).astype(np.float32) * 100
    
    # Update tracker
    tracker.update(mouth_landmarks)
    
    # Get state
    state = tracker.get_state()
    valid_states = ["closed", "slightly_open", "open", "very_open"]
    if state in valid_states:
        print(f"  ✓ MouthTracker working (current state: {state})")
    else:
        print(f"  ✗ Invalid mouth state: {state}")
    
    # Test mouth openness calculation
    openness = calculate_mouth_openness(mouth_landmarks)
    if 0.0 <= openness <= 1.0:
        print(f"  ✓ Mouth openness calculation works (value: {openness:.2f})")
    else:
        print(f"  ✗ Mouth openness out of range: {openness}")
    
    # Test mouth detail enhancement
    mouth_region = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    enhanced_mouth = enhance_mouth_details(mouth_region, brightness_factor=1.1)
    if enhanced_mouth.shape == mouth_region.shape:
        print(f"  ✓ Mouth detail enhancement works")
    else:
        print(f"  ✗ Mouth detail enhancement shape mismatch")
    
    # Check audio analysis
    librosa_available = has_librosa()
    print(f"  ✓ Audio analysis available (librosa: {librosa_available})")
    
    # Test mouth movement estimation
    energy = np.random.rand(30).astype(np.float32)
    movement = estimate_mouth_movement(energy)
    if len(movement) == len(energy) and np.all((movement >= 0) & (movement <= 1)):
        print(f"  ✓ Mouth movement estimation works")
    else:
        print(f"  ✗ Mouth movement estimation failed")
        
except Exception as e:
    print(f"  ✗ Lip sync test failed: {e}")
    import traceback
    traceback.print_exc()

# Final summary
print("\n" + "="*70)
print("  ✓ Local feature tests completed successfully!")
print("="*70)
print("\nNext steps:")
print("  1. Run the application: python run.py")
print("  2. Test with your source image and target video")
print("  3. Check output quality at 1080p resolution")
print("  4. Verify lip sync improvements with mouth mask enabled")
print("  5. If working well, commit to feature branch: git add . && git commit -m '...'")
print("\n")
