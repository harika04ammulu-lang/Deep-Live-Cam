# Deep-Live-Cam Enhancement Quick Reference

## ⚡ Quick Enable

```python
import modules.globals

# Enable all enhancements
modules.globals.enable_expression_transfer = True
modules.globals.enable_mouth_morphing = True
modules.globals.enable_landmark_smoothing = True

# Use Kalman filter for superior smoothing
modules.globals.use_kalman_filter = True
modules.globals.enable_bidirectional_filter = True

# Process
from modules.processors.frame.face_swapper import process_video
process_video("source.jpg", "target.mp4", "output.mp4")
```

---

## 📋 Configuration Cheat Sheet

| Feature | Enable | Strength | Smoothing |
|---------|--------|----------|-----------|
| **Expression** | `enable_expression_transfer` | `expression_transfer_strength` (0.7) | `expression_smoothing` (0.7) |
| **Mouth Morph** | `enable_mouth_morphing` | `mouth_morph_strength` (0.7) | `mouth_openness_smoothing` (0.8) |
| **Landmarks (EMA)** | `enable_landmark_smoothing` | N/A | `mouth_smoothing_factor` (0.85) |
| **Landmarks (Kalman)** | `use_kalman_filter` | N/A | `enable_bidirectional_filter` (True) |

---

## 🎯 Presets

### High Quality (Slower)
```python
modules.globals.enable_expression_transfer = True
modules.globals.expression_transfer_strength = 0.9
modules.globals.enable_mouth_morphing = True
modules.globals.mouth_morph_strength = 0.9
modules.globals.use_kalman_filter = True
modules.globals.enable_bidirectional_filter = True
```

### Balanced (Recommended)
```python
modules.globals.enable_expression_transfer = True
modules.globals.expression_transfer_strength = 0.7
modules.globals.enable_mouth_morphing = True
modules.globals.mouth_morph_strength = 0.7
modules.globals.use_kalman_filter = True
```

### Performance (Faster)
```python
modules.globals.enable_expression_transfer = False
modules.globals.enable_mouth_morphing = True
modules.globals.mouth_morph_strength = 0.6
modules.globals.use_kalman_filter = False  # Use EMA instead
```

---

## 🐛 Debug Mode

```python
# Show what's being detected
modules.globals.show_expression_overlay = True
modules.globals.show_landmark_debug_overlay = True
```

---

## 📦 Installation

```bash
pip install mediapipe>=0.10.0
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No expression | Install MediaPipe |
| Too weak | Increase `*_strength` to 0.9 |
| Too strong | Decrease `*_strength` to 0.5 |
| Jittery | Increase `*_smoothing` to 0.9 |
| Laggy | Decrease `*_smoothing` to 0.6 |

---

## 📚 Documentation

- `COMPLETE_ENHANCEMENT_SUMMARY.md` - Overview of all features
- `EXPRESSION_TRANSFER_GUIDE.md` - Expression transfer details
- `MOUTH_MORPHING_INTEGRATION.md` - Mouth morphing details
- `KALMAN_FILTER_GUIDE.md` - Kalman filter landmark smoothing

---

## ✅ Test

```bash
python test_expression_transfer.py
python test_mouth_morphing.py
python test_kalman_filter.py
```

---

**That's it! 🎉 Enjoy more natural face swapping!**
