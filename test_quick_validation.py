"""
Quick Validation Test - Tests module imports and syntax without dependencies

This validates that all modules can be imported and have correct structure.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

TESTS_PASSED = 0
TESTS_FAILED = 0


def test_passed(name):
    global TESTS_PASSED
    TESTS_PASSED += 1
    print(f"✓ {name}")


def test_failed(name, error):
    global TESTS_FAILED
    TESTS_FAILED += 1
    print(f"✗ {name}: {error}")


def main():
    print("\n" + "="*70)
    print("  Quick Validation Test - Module Imports & Structure")
    print("="*70 + "\n")
    
    # Test 1: Module imports
    modules_to_test = [
        ("modules.globals", ["quality_mode", "target_resolution", "enable_lip_sync"]),
        ("modules.video_capture", ["VideoCapturer", "RESOLUTION_PRESETS"]),
        ("modules.processors.frame.quality_enhancer", ["upscale_frame", "apply_quality_profile"]),
        ("modules.processors.frame.lip_sync_enhancer", ["MouthTracker", "calculate_mouth_openness"]),
        ("modules.processors.frame.audio_analysis", ["has_librosa", "estimate_mouth_movement"]),
    ]
    
    print("Testing Module Imports:\n")
    for module_name, expected_items in modules_to_test:
        try:
            module = __import__(module_name, fromlist=expected_items)
            
            # Check for expected items
            missing = []
            for item in expected_items:
                if not hasattr(module, item):
                    missing.append(item)
            
            if missing:
                test_failed(f"Import {module_name}", f"Missing: {missing}")
            else:
                test_passed(f"Import {module_name}")
                
        except Exception as e:
            test_failed(f"Import {module_name}", str(e)[:60])
    
    # Test 2: File existence
    print("\n\nTesting File Existence:\n")
    files_to_check = [
        "modules/processors/frame/quality_enhancer.py",
        "modules/processors/frame/lip_sync_enhancer.py",
        "modules/processors/frame/audio_analysis.py",
        "QUALITY_AND_LIP_SYNC_ENHANCEMENTS.md",
        "integration_example.py",
    ]
    
    for filepath in files_to_check:
        full_path = Path(filepath)
        if full_path.exists():
            size = full_path.stat().st_size / 1024  # KB
            test_passed(f"File {filepath} ({size:.1f}KB)")
        else:
            test_failed(f"File {filepath}", "Not found")
    
    # Test 3: Check for syntax errors by compiling
    print("\n\nTesting Python Syntax:\n")
    import py_compile
    
    python_files = [
        "modules/processors/frame/quality_enhancer.py",
        "modules/processors/frame/lip_sync_enhancer.py",
        "modules/processors/frame/audio_analysis.py",
        "integration_example.py",
        "test_enhancements.py",
    ]
    
    for filepath in python_files:
        try:
            py_compile.compile(filepath, doraise=True)
            test_passed(f"Syntax {filepath}")
        except py_compile.PyCompileError as e:
            test_failed(f"Syntax {filepath}", str(e)[:60])
    
    # Summary
    print("\n" + "="*70)
    print(f"  Results: {TESTS_PASSED} passed, {TESTS_FAILED} failed")
    print("="*70 + "\n")
    
    if TESTS_FAILED == 0:
        print("✓ All validation tests passed! Code is ready for local testing.\n")
        return 0
    else:
        print(f"✗ {TESTS_FAILED} validation issue(s) found.\n")
        return 1


if __name__ == "__main__":
    exit(main())
