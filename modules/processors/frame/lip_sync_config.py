"""
Lip Sync Configuration & Fine-tuning

Advanced configuration options for realistic mouth movements and lip sync.
"""

import numpy as np
from typing import Dict, Any

# Mouth Movement Configuration
MOUTH_MOVEMENT_CONFIG = {
    # Shape transition settings
    "transition_frames": 5,           # Frames for smooth shape transition
    "shape_hold_frames": 3,           # Frames to hold shape before transition
    
    # Jaw movement
    "enable_jaw_movement": True,      # Sync jaw with audio energy
    "jaw_sensitivity": 0.8,           # How much jaw responds to audio (0.0-1.0)
    "jaw_smoothing": 0.7,             # Jaw movement smoothing (0.0-1.0)
    "max_jaw_offset": 15,             # Maximum jaw movement in pixels
    
    # Blinking
    "enable_eye_blinks": True,        # Synchronize eye blinks
    "blink_probability": 0.03,        # Probability per frame (0.0-1.0)
    "blink_frames": 3,                # Frames per blink cycle
    
    # Temporal smoothing
    "enable_temporal_smoothing": True,
    "smoothing_window": 3,            # Frames for averaging
    "smoothing_weight": 0.8,          # Weight for current frame vs history
    
    # Mouth detail enhancement
    "enable_detail_enhancement": True,
    "enhancement_strength": 0.7,      # Detail enhancement strength (0.0-1.0)
    "enable_color_correction": True,  # Match mouth color to face
    
    # Confidence-based blending
    "enable_confidence_blending": True,
    "min_confidence": 0.3,            # Minimum confidence for blending
    "confidence_influence": 0.8,      # How much confidence affects blend
    
    # Advanced mouth morphing
    "enable_mouth_morphing": True,
    "morphing_interpolation": "ease_in_out",  # ease_in_out, linear, smooth
    "morphing_frames": 5,             # Frames for morphing between shapes
    
    # Deformation strength
    "mouth_deformation_strength": 0.5,  # Strength of shape deformation
    "enable_optical_flow": False,       # Enable optical flow-based warping
    
    # Audio-visual sync
    "audio_sync_delay": 0,            # Frame delay for audio-visual sync
    "energy_to_shape_mapping": {      # Map energy to mouth shape
        (0.0, 0.1): "closed",
        (0.1, 0.3): "neutral",
        (0.3, 0.5): "open_e",
        (0.5, 0.8): "open_a",
        (0.8, 1.0): "open_a",
    }
}

# Mouth shape morphing reference points
MOUTH_MORPHING_POINTS = {
    "corners": [0, 16],           # Left and right mouth corners
    "center_top": [7],             # Top center
    "center_bottom": [13],         # Bottom center
    "upper_lip": [2, 3, 4],        # Upper lip key points
    "lower_lip": [10, 11, 12],     # Lower lip key points
    "inner_top": [14],             # Inner upper lip
    "inner_bottom": [15],          # Inner lower lip
}

# Shape-specific parameters for realistic animation
SHAPE_ANIMATION_PARAMS = {
    "closed": {
        "lip_compression": 0.9,      # How much lips are compressed
        "jaw_position": -0.2,         # Jaw slightly closed
        "mouth_width": 0.8,           # Mouth width relative to normal
        "inner_opening": 0.0,         # No inner mouth visibility
        "blending_strength": 0.9,     # Strong blending
    },
    "neutral": {
        "lip_compression": 0.7,
        "jaw_position": 0.0,
        "mouth_width": 1.0,
        "inner_opening": 0.1,
        "blending_strength": 0.8,
    },
    "open_e": {
        "lip_compression": 0.5,
        "jaw_position": 0.3,
        "mouth_width": 1.1,
        "inner_opening": 0.3,
        "blending_strength": 0.7,
    },
    "open_a": {
        "lip_compression": 0.2,
        "jaw_position": 0.8,
        "mouth_width": 1.3,
        "inner_opening": 0.6,
        "blending_strength": 0.6,
    },
    "rounded": {
        "lip_compression": 0.4,
        "jaw_position": 0.2,
        "mouth_width": 0.7,
        "inner_opening": 0.2,
        "blending_strength": 0.75,
    },
}

# Phoneme to viseme mapping (more accurate)
PHONEME_TO_VISEME = {
    # Vowels - organized by articulation height
    "AA": "open_a",   # /ɑ:/ - father, palm (very open)
    "AE": "open_a",   # /æ/ - trap, cat (open)
    "AH": "open_a",   # /ʌ/ - strut, cup (open)
    "AO": "open_a",   # /ɔ:/ - thought, force (open-rounded)
    "AW": "rounded",  # /aʊ/ - mouth, now (rounded)
    "AY": "open_e",   # /aɪ/ - price, try (diphthong)
    "ER": "neutral",  # /ɜ:/ - nurse, turn (neutral)
    "EH": "open_e",   # /ɛ/ - dress, bed (open-front)
    "EY": "open_e",   # /eɪ/ - face, day (open-front)
    "IH": "open_e",   # /ɪ/ - kit, bid (front-closed)
    "IY": "open_e",   # /i:/ - fleece, see (front-closed)
    "OW": "rounded",  # /oʊ/ - goat, go (rounded)
    "OY": "rounded",  # /ɔɪ/ - choice, boy (rounded)
    "UH": "rounded",  # /ʊ/ - foot, good (rounded)
    "UW": "rounded",  # /u:/ - goose, who (rounded-closed)
    
    # Consonants - lip-related
    "B": "closed",    # /b/ - bilabial stop (lips together)
    "M": "closed",    # /m/ - bilabial nasal (lips together)
    "P": "closed",    # /p/ - bilabial stop (lips together)
    "F": "open_e",    # /f/ - labiodental (lower lip out)
    "V": "open_e",    # /v/ - labiodental (lower lip out)
    "W": "rounded",   # /w/ - labial approximant (rounded)
    
    # Other consonants (neutral default)
    "D": "neutral",
    "G": "neutral",
    "K": "neutral",
    "L": "neutral",
    "N": "closed",    # /n/ - alveolar nasal
    "R": "rounded",   # /r/ - approximant
    "S": "open_e",    # /s/ - sibilant (slight opening)
    "T": "neutral",
    "Z": "open_e",    # /z/ - sibilant
    "CH": "open_e",   # /tʃ/ - affricate
    "JH": "open_e",   # /dʒ/ - affricate
    "SH": "rounded",  # /ʃ/ - postalveolar
    "TH": "open_e",   # /θ/ - dental fricative
    "ZH": "open_e",   # /ʒ/ - postalveolar
    "NG": "neutral",  # /ŋ/ - velar nasal
    "HH": "neutral",  # /h/ - glottal fricative
    "Y": "open_e",    # /j/ - palatal approximant
}

def get_mouth_config() -> Dict[str, Any]:
    """Get current mouth movement configuration."""
    return MOUTH_MOVEMENT_CONFIG.copy()

def set_mouth_config(key: str, value: Any) -> bool:
    """
    Update mouth movement configuration.
    
    Args:
        key: Configuration key
        value: New value
        
    Returns:
        True if successful, False otherwise
    """
    if key in MOUTH_MOVEMENT_CONFIG:
        MOUTH_MOVEMENT_CONFIG[key] = value
        return True
    return False

def get_shape_params(shape: str) -> Dict[str, float]:
    """Get animation parameters for a specific mouth shape."""
    return SHAPE_ANIMATION_PARAMS.get(shape, SHAPE_ANIMATION_PARAMS["neutral"]).copy()

def get_phoneme_shape(phoneme: str) -> str:
    """Get mouth shape for a phoneme."""
    return PHONEME_TO_VISEME.get(phoneme, "neutral")

# Quick preset configurations for different quality levels
QUALITY_PRESETS = {
    "low": {
        "enable_temporal_smoothing": True,
        "smoothing_window": 5,
        "enable_jaw_movement": False,
        "enable_eye_blinks": False,
        "enable_detail_enhancement": False,
        "enable_mouth_morphing": False,
    },
    "medium": {
        "enable_temporal_smoothing": True,
        "smoothing_window": 3,
        "enable_jaw_movement": True,
        "enable_eye_blinks": True,
        "enable_detail_enhancement": True,
        "enable_mouth_morphing": True,
        "morphing_frames": 5,
    },
    "high": {
        "enable_temporal_smoothing": True,
        "smoothing_window": 3,
        "enable_jaw_movement": True,
        "enable_eye_blinks": True,
        "enable_detail_enhancement": True,
        "enable_mouth_morphing": True,
        "morphing_frames": 5,
        "enable_optical_flow": False,
        "enable_confidence_blending": True,
    },
    "ultra": {
        "enable_temporal_smoothing": True,
        "smoothing_window": 2,
        "enable_jaw_movement": True,
        "enable_eye_blinks": True,
        "enable_detail_enhancement": True,
        "enable_mouth_morphing": True,
        "morphing_frames": 8,
        "enable_optical_flow": True,
        "enable_confidence_blending": True,
        "enhancement_strength": 0.9,
    },
}

def apply_preset(preset_name: str) -> bool:
    """
    Apply a quality preset to mouth configuration.
    
    Args:
        preset_name: Preset name (low, medium, high, ultra)
        
    Returns:
        True if successful, False otherwise
    """
    if preset_name not in QUALITY_PRESETS:
        return False
    
    preset = QUALITY_PRESETS[preset_name]
    for key, value in preset.items():
        if key in MOUTH_MOVEMENT_CONFIG:
            MOUTH_MOVEMENT_CONFIG[key] = value
    
    return True
