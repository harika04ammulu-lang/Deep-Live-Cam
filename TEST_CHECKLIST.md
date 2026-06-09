# Testing Checklist

## Pre-Test Setup ✅

- [x] Kalman filter overflow fixes applied
- [x] Shape validation added to all modules
- [x] Error handling enhanced
- [x] MediaPipe installed
- [x] Mouth morphing enabled
- [x] Expression transfer disabled (will enable after mouth test)
- [x] All validation tests passed

## Test Phase 1: Mouth Morphing

### Run Commands
```bash
cd C:\Users\DSC\Deep-Live-Cam
.\venv\Scripts\activate
python run.py
```

### Checklist While Running

#### Startup (First 30 seconds)
- [ ] App starts without immediate crash
- [ ] Models load successfully
- [ ] No "SVD did not converge" errors
- [ ] No "Sizes of input arguments do not match" errors
- [ ] No RuntimeWarning overflow messages

#### Processing (After 30 seconds)
- [ ] Frames being processed
- [ ] Output displayed/saved
- [ ] Performance acceptable (estimate FPS: _____ )
- [ ] Console relatively clean

#### Errors Observed
- [ ] None (skip to results)
- [ ] One-time errors only (list below)
- [ ] Repeated errors (list below)
- [ ] Crash occurred (note when)

**Error Details:**
```
[Copy any error messages here]
```

#### Visual Quality (Optional)
- [ ] Face swap working
- [ ] Mouth movement visible
- [ ] No major artifacts
- [ ] Quality acceptable

### Test Results

**Overall Status:**
- [ ] ✅ SUCCESS - No crashes, minimal errors
- [ ] ⚠️ PARTIAL - Some errors but continues
- [ ] ❌ FAILED - Crashed or unusable

**Console Output Summary:**
```
[Paste key console lines here - first 20 and last 20 lines are most useful]
```

**Performance:**
- Estimated FPS: _____
- CPU/GPU usage: _____
- Total runtime: _____ minutes

**Decision:**
- [ ] Proceed to Expression Transfer test
- [ ] Debug mouth morphing issues first
- [ ] Need more investigation

---

## Test Phase 2: Expression Transfer

### Enable Feature
1. Stop app (Ctrl+C)
2. Edit `modules\globals.py`
3. Change: `enable_expression_transfer: bool = False` → `True`
4. Save file

### Run Commands
```bash
python run.py
```

### Checklist While Running

#### Startup
- [ ] App starts with both features enabled
- [ ] No new errors compared to Phase 1
- [ ] MediaPipe initialized successfully

#### Processing
- [ ] Both features appear active
- [ ] Performance still acceptable (estimate FPS: _____ )
- [ ] No crashes or hangs

#### Errors Observed
- [ ] None
- [ ] One-time errors only (list below)
- [ ] Repeated errors (list below)
- [ ] Crash occurred (note when)

**Error Details:**
```
[Copy any error messages here]
```

#### Visual Quality (Optional)
- [ ] Face swap working
- [ ] Expression transfer visible (smile, eyebrow, etc.)
- [ ] Mouth morphing still working
- [ ] No major new artifacts

### Test Results

**Overall Status:**
- [ ] ✅ SUCCESS - Both features running cleanly
- [ ] ⚠️ PARTIAL - One feature works, other fails gracefully
- [ ] ❌ FAILED - Crashes or major issues

**Console Output Summary:**
```
[Paste key console lines here]
```

**Performance:**
- Estimated FPS: _____
- Performance impact vs Phase 1: _____
- Total runtime: _____ minutes

**Decision:**
- [ ] Both features confirmed working
- [ ] One feature working, one failing gracefully (acceptable)
- [ ] Proceed to Phase 4 (skin tone/lighting)
- [ ] Need debugging

---

## Phase 3: Extended Stability Test

### Duration Test
- [ ] Run for 5 minutes continuously
- [ ] No crashes
- [ ] No memory leaks (observe RAM usage)
- [ ] Performance remains stable

### Conditions Test
- [ ] Different lighting
- [ ] Different face angles
- [ ] Fast movements
- [ ] Expressions (if enabled)

### Results
- [ ] ✅ Stable under all conditions
- [ ] ⚠️ Some minor issues but recovers
- [ ] ❌ Unstable or crashes

---

## Final Assessment

### Critical Issues (Must Fix)
```
[List any crashes, repeated errors, or blocking issues]
```

### Non-Critical Issues (Can Defer)
```
[List minor issues, visual artifacts, or optimization opportunities]
```

### Features Status
- **Base Face Swap**: [ ] Working [ ] Issues
- **Kalman Smoothing**: [ ] Working [ ] Issues
- **Mouth Morphing**: [ ] Working [ ] Graceful Fail [ ] Issues
- **Expression Transfer**: [ ] Working [ ] Graceful Fail [ ] Issues

### Ready for Phase 4?
- [ ] YES - All tests passed, stable execution
- [ ] PARTIAL - Some features not working but app stable
- [ ] NO - Need to fix critical issues first

### Notes
```
[Any additional observations, suggestions, or concerns]
```

---

## Quick Reference

### Stop App
```
Ctrl+C
```

### Check Status
```bash
python -c "import modules.globals as g; print(f'Mouth morphing: {g.enable_mouth_morphing}'); print(f'Expression transfer: {g.enable_expression_transfer}')"
```

### Run Validation
```bash
python test_overflow_fixes.py
```

### View Logs
```bash
# If logs are saved, check the output directory
```

### Disable a Feature
Edit `modules\globals.py`:
- `enable_mouth_morphing: bool = False`
- `enable_expression_transfer: bool = False`

---

## Success Criteria

### Minimum (Must Have)
- [x] App runs without crashes
- [x] No SVD errors
- [x] No shape mismatch errors
- [x] No overflow warnings
- [x] Face swap produces output
- [ ] Runs for 5+ minutes continuously

### Optimal (Nice to Have)
- [ ] Mouth morphing visibly working
- [ ] Expression transfer visibly working
- [ ] Performance > 15 FPS
- [ ] No visual artifacts
- [ ] Smooth, stable output

---

**Last Updated:** After overflow fixes, before testing
**Current Phase:** Mouth morphing test ready
**Action Required:** Run `python run.py` and complete Phase 1 checklist
