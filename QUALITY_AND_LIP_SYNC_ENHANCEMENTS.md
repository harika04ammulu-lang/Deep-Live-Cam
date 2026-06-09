# Deep-Live-Cam 1080P+ Quality and Enhanced Lip Sync Enhancement

## Overview

This feature branch introduces significant quality improvements and advanced lip sync features for Deep-Live-Cam, enabling:

- **1080P+ Video Output**: Full support for 1080p (1920x1080), 1440p (2560x1440), and 2K/4K resolutions
- **Super-Resolution Upscaling**: Intelligent upscaling for lower-quality source videos
- **Enhanced Lip Sync**: Advanced mouth detection, tracking, and synchronization
- **Audio Analysis**: Optional audio-based mouth shape prediction
- **Quality Enhancement**: Frame enhancement with detail preservation

## New Features

### 1. High-Resolution Video Capture

#### Resolution Presets

The `VideoCapturer` class now supports predefined resolution presets:

- **480p**: 854x480 (low bandwidth)
- **720p**: 1280x720 (standard HD)
- **1080p**: 1920x1080 (Full HD) **[NEW DEFAULT]**
- **1440p**: 2560x1440 (2K)
- **2K**: 2560x1440 (2K)
- **4K**: 3840x2160 (Ultra HD)

#### Usage

```python
from modules.video_capture import VideoCapturer

# Create capturer
capturer = VideoCapturer(device_index=0)

# Start with 1080p preset
capturer.start_with_preset("1080p", fps=60)

# Or use custom resolution
capturer.start(width=1920, height=1080, fps=60)

# Get available presets
presets = capturer.get_available_resolutions()
print(presets)  # {'480p': (854, 480), '720p': (1280, 720), ...}
```

### 2. Quality Enhancement Module

New module: `modules/processors/frame/quality_enhancer.py`

#### Features

- **Upscaling**: Multiple interpolation methods (Lanczos, Cubic, Linear)
- **Detail Enhancement**: Unsharp masking for sharper output
- **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Adaptive Quality**: Automatically adjusts enhancement based on performance

#### Quality Presets

```python
from modules.processors.frame.quality_enhancer import get_quality_preset

# Standard (720p)
standard = get_quality_preset("standard")
# {'resolution': (1280, 720), 'upscale_factor': 1.0, ...}

# High (1080p)
high = get_quality_preset("high")
# {'resolution': (1920, 1080), 'upscale_factor': 1.25, ...}

# Ultra (2K+)
ultra = get_quality_preset("ultra")
# {'resolution': (3840, 2160), 'upscale_factor': 1.5, ...}
```

#### Example Usage

```python
from modules.processors.frame.quality_enhancer import (
    upscale_frame, 
    enhance_frame_detail,
    apply_quality_profile
)

# Upscale a frame by 1.5x with Lanczos interpolation
upscaled = upscale_frame(frame, factor=1.5, method="lanczos")

# Enhance detail sharpness
enhanced = enhance_frame_detail(upscaled, strength=0.7)

# Apply complete quality profile
result = apply_quality_profile(frame, profile="high")
```

### 3. Enhanced Lip Sync Module

New module: `modules/processors/frame/lip_sync_enhancer.py`

#### Features

- **Mouth Tracking**: Temporal smoothing for natural motion
- **Mouth State Classification**: Closed, Slightly Open, Open, Very Open
- **Mouth Shape Analysis**: Calculate mouth openness from landmarks
- **Improved Blending**: Feathered blending for smooth transitions
- **Detail Enhancement**: Enhance mouth region colors and details

#### Mouth Tracking

```python
from modules.processors.frame.lip_sync_enhancer import MouthTracker

# Create tracker
tracker = MouthTracker(history_size=5)

# Update with new landmarks
tracker.update(mouth_landmarks)

# Get smoothed landmarks
smoothed = tracker.get_smoothed_landmarks()

# Get mouth state
state = tracker.get_state()  # 'closed', 'slightly_open', 'open', 'very_open'
```

#### Mouth Openness Calculation

```python
from modules.processors.frame.lip_sync_enhancer import calculate_mouth_openness

# Calculate mouth openness (0.0=closed, 1.0=very open)
openness = calculate_mouth_openness(mouth_landmarks)
```

#### Improved Mouth Blending

```python
from modules.processors.frame.lip_sync_enhancer import blend_mouth_region

# Blend mouth region with smooth feathering
result = blend_mouth_region(
    frame,
    enhanced_mouth_region,
    original_mouth_region,
    bbox=(x_min, y_min, x_max, y_max),
    blend_strength=0.8,
    feather_size=25
)
```

### 4. Audio Analysis Module

New module: `modules/processors/frame/audio_analysis.py`

#### Features

- **Audio Energy Extraction**: Analyzes volume and energy for mouth movement
- **Voice Activity Detection**: Identifies speech regions
- **Mouth Shape Prediction**: Audio-based mouth shape estimation
- **Optional Librosa Support**: Enhanced features with librosa if available

#### Audio Energy Analysis

```python
from modules.processors.frame.audio_analysis import get_audio_energy

# Extract audio from video
from modules.processors.frame.audio_analysis import extract_audio_from_video
extract_audio_from_video("video.mp4", "audio.wav")

# Get energy envelope
energy = get_audio_energy("audio.wav", frame_rate=30)
```

#### Voice Activity Detection

```python
from modules.processors.frame.audio_analysis import get_voice_activity

# Detect speech regions
vad = get_voice_activity("audio.wav", frame_rate=30, threshold=0.1)
# Returns binary array: 1=speaking, 0=silence
```

#### Mouth Shape Prediction

```python
from modules.processors.frame.audio_analysis import predict_mouth_shape_from_audio

# Predict mouth shapes from audio
mouth_shapes = predict_mouth_shape_from_audio("audio.wav", frame_rate=30)
# Returns array: [0=closed, 1=slightly_open, 2=open, 3=very_open]
```

## Global Configuration

New settings in `modules/globals.py`:

### Quality Settings

```python
# Quality Modes: "standard" (720p), "high" (1080p), "ultra" (1440p+)
quality_mode: str = "high"

# Target output resolution (width, height)
target_resolution: tuple = (1920, 1080)

# Enable super-resolution upscaling for lower-quality sources
enable_upscaling: bool = True

# Upscaling multiplier (1.0=none, 1.5=1.5x, 2.0=2x)
upscaling_factor: float = 1.5

# Enable GFPGAN face enhancement
enable_face_enhancement: bool = True

# Strength of face enhancement (0.0-1.0)
face_enhancement_strength: float = 0.7

# Dynamically adjust quality based on performance
adaptive_quality: bool = True
```

### Lip Sync Settings

```python
# Enable advanced lip sync features
enable_lip_sync: bool = True

# Strength of lip sync effect (0.0-1.0)
lip_sync_strength: float = 0.8

# Use detailed mouth landmarks for better tracking
enable_mouth_landmarks: bool = True

# Feathering for mouth blending (higher = smoother)
mouth_blend_feather: int = 25

# Analyze audio for mouth shape matching (requires librosa)
enable_audio_analysis: bool = False

# Threshold for audio-based mouth movement
audio_sync_threshold: float = 0.5
```

## Integration Guide

### 1. Using Quality Enhancement in Video Processing

```python
from modules.processors.frame.quality_enhancer import apply_quality_profile, upscale_frame

def process_frame(frame, quality_mode="high"):
    # Upscale if needed
    if frame.shape[0] < 1080:
        frame = upscale_frame(frame, factor=1.5)
    
    # Apply quality profile
    frame = apply_quality_profile(frame, quality_mode)
    
    return frame
```

### 2. Using Enhanced Lip Sync

```python
from modules.processors.frame.lip_sync_enhancer import (
    MouthTracker,
    get_mouth_region,
    blend_mouth_region,
    enhance_mouth_details
)

# In face processing loop:
mouth_tracker = MouthTracker()

for face in faces:
    # Get mouth region
    mouth_region, bbox = get_mouth_region(face, frame)
    
    if mouth_region is not None:
        # Extract landmarks
        mouth_landmarks = face.landmark_2d_106[48:68]
        
        # Update tracker for smoothing
        mouth_tracker.update(mouth_landmarks)
        
        # Enhance mouth details
        enhanced_mouth = enhance_mouth_details(mouth_region)
        
        # Blend back with feathering
        frame = blend_mouth_region(
            frame,
            enhanced_mouth,
            mouth_region,
            bbox,
            blend_strength=0.8,
            feather_size=25
        )
```

### 3. Using Audio-Based Lip Sync

```python
from modules.processors.frame.audio_analysis import (
    extract_audio_from_video,
    predict_mouth_shape_from_audio
)

# Extract audio from video
extract_audio_from_video("video.mp4", "audio.wav")

# Predict mouth shapes based on audio
mouth_shapes = predict_mouth_shape_from_audio("audio.wav", frame_rate=30)

# Use mouth_shapes to guide mouth region manipulation
for frame_idx, mouth_shape in enumerate(mouth_shapes):
    # Apply appropriate mouth modifications based on predicted shape
    # shape: 0=closed, 1=slightly_open, 2=open, 3=very_open
    pass
```

## Performance Recommendations

### GPU Requirements

- **1080p at 60fps**: NVIDIA RTX 2060+ recommended
- **1440p at 60fps**: NVIDIA RTX 3060 Ti+ recommended  
- **2K/4K**: NVIDIA RTX 3080+ recommended

### CPU Requirements

For real-time processing without GPU:
- 1080p @ 30fps: AMD Ryzen 9 5900X+ or Intel i9-11900K+
- Enable `adaptive_quality` for dynamic quality adjustment

### Memory Requirements

- Standard (720p): 4GB RAM minimum
- High (1080p): 8GB RAM minimum
- Ultra (1440p+): 16GB RAM minimum

## Troubleshooting

### Issue: Output resolution doesn't match target

**Solution**: Not all cameras support all resolutions. Use:
```python
capturer.start_with_preset("1080p")  # Fallback to supported resolution
```

### Issue: Performance degradation with high resolution

**Solution**: Enable adaptive quality:
```python
import modules.globals
modules.globals.adaptive_quality = True
modules.globals.quality_mode = "high"
```

### Issue: Mouth region looks blurry

**Solution**: Increase enhancement strength:
```python
modules.globals.face_enhancement_strength = 0.9
modules.globals.enable_upscaling = True
```

## Dependencies

### Required
- OpenCV (cv2) - core image processing
- NumPy - numerical operations
- existing Deep-Live-Cam dependencies

### Optional
- librosa - enhanced audio analysis for better lip sync
- soundfile - audio file handling (installed with librosa)

### Installation

For audio analysis features:
```bash
pip install librosa soundfile
```

## Testing

### Unit Tests

```bash
python -m pytest tests/test_quality_enhancer.py
python -m pytest tests/test_lip_sync_enhancer.py
python -m pytest tests/test_audio_analysis.py
```

### Integration Testing

1. Test video capture with different resolutions
2. Test quality enhancement pipeline
3. Test lip sync with various mouth masks
4. Test audio analysis with sample videos

## Future Enhancements

1. **Real-time GFPGAN Integration**: GPU-accelerated face enhancement
2. **Advanced Phoneme Recognition**: Better audio-based mouth prediction
3. **Motion Interpolation**: Frame interpolation for ultra-smooth video
4. **Adaptive Bitrate**: Dynamic bitrate adjustment based on content
5. **Multi-face Quality Optimization**: Per-face quality settings

## Contributing

When adding new features:

1. Add tests in appropriate test file
2. Update this README with new features
3. Follow existing code style and threading patterns
4. Use locks for thread-safe operations
5. Document all public functions

## License

Same as Deep-Live-Cam parent project
