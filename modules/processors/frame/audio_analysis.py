"""
Audio Analysis Module for Lip Sync Enhancement

Analyzes audio features to improve mouth synchronization and lip sync quality.
Supports basic audio analysis without external dependencies, with optional
advanced features if librosa is available.

Features:
- Audio feature extraction (energy, MFCC, spectral features)
- Voice activity detection
- Phoneme-based mouth shape prediction
- Audio-visual synchronization
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import subprocess
import os
import threading

AUDIO_ANALYSIS_LOCK = threading.Lock()

# Audio feature cache
_AUDIO_CACHE = {}

# Phoneme to mouth shape mapping
PHONEME_TO_MOUTH_SHAPE = {
    # Vowels
    "AA": "open",  # open
    "AE": "open",  # open
    "AH": "open",  # open
    "AO": "open",  # open
    "AW": "open",  # open
    "AY": "open",  # open
    "ER": "slightly_open",  # neutral
    "EY": "open",  # open
    "IH": "slightly_open",  # neutral
    "IY": "slightly_open",  # neutral
    "OW": "open",  # open
    "OY": "open",  # open
    "UH": "closed",  # closed
    "UW": "closed",  # closed
    # Consonants - labial (lip-relevant)
    "B": "closed",  # bilabial stop
    "M": "closed",  # bilabial nasal
    "P": "closed",  # bilabial stop
    "F": "slightly_open",  # labiodental
    "V": "slightly_open",  # labiodental
    "W": "slightly_open",  # labial approximant
    # Other consonants
    "D": "slightly_open",  # alveolar
    "G": "slightly_open",  # velar
    "K": "slightly_open",  # velar
    "L": "slightly_open",  # alveolar
    "N": "closed",  # alveolar nasal
    "R": "slightly_open",  # alveolar approximant
    "S": "slightly_open",  # alveolar fricative
    "T": "slightly_open",  # alveolar stop
    "Z": "slightly_open",  # alveolar fricative
    "CH": "slightly_open",  # postalveolar affricate
    "JH": "slightly_open",  # postalveolar affricate
    "SH": "slightly_open",  # postalveolar fricative
    "TH": "slightly_open",  # dental fricative
    "ZH": "slightly_open",  # postalveolar fricative
    "NG": "closed",  # velar nasal
    "HH": "slightly_open",  # glottal
    "Y": "slightly_open",  # palatal
}


def has_librosa() -> bool:
    """Check if librosa is available."""
    try:
        import librosa
        return True
    except ImportError:
        return False


def extract_audio_from_video(video_path: str, audio_path: str = "temp_audio.wav") -> bool:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file
        audio_path: Path to save extracted audio

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-q:a", "9",
            "-n",  # Don't overwrite existing files
            audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(audio_path)
    except Exception as e:
        print(f"[AudioAnalysis] Failed to extract audio: {e}")
        return False


def get_audio_energy(
    audio_path: str,
    frame_rate: int = 30,
    n_fft: int = 2048
) -> Optional[np.ndarray]:
    """
    Extract energy envelope from audio.

    Args:
        audio_path: Path to audio file
        frame_rate: Frame rate for alignment
        n_fft: FFT window size

    Returns:
        Energy array aligned with video frames, or None if unavailable
    """
    try:
        if has_librosa():
            return _get_audio_energy_librosa(audio_path, frame_rate, n_fft)
        else:
            return _get_audio_energy_basic(audio_path, frame_rate)
    except Exception as e:
        print(f"[AudioAnalysis] Error extracting audio energy: {e}")
        return None


def _get_audio_energy_librosa(
    audio_path: str,
    frame_rate: int = 30,
    n_fft: int = 2048
) -> np.ndarray:
    """Extract energy using librosa."""
    import librosa

    y, sr = librosa.load(audio_path, sr=None)
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    energy = np.sqrt(np.mean(S**2, axis=0))

    # Resample energy to match video frame rate
    num_frames = int(len(y) / sr * frame_rate)
    energy_resampled = np.interp(
        np.linspace(0, len(energy) - 1, num_frames),
        np.arange(len(energy)),
        energy
    )

    return energy_resampled / np.max(energy_resampled)


def _get_audio_energy_basic(
    audio_path: str,
    frame_rate: int = 30
) -> np.ndarray:
    """Extract energy using basic analysis without librosa."""
    import wave
    import struct

    try:
        with wave.open(audio_path, 'rb') as wav_file:
            n_frames = wav_file.getnframes()
            n_channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frame_width = wav_file.getsampwidth()

            # Read audio data
            audio_data = wav_file.readframes(n_frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Reshape for multi-channel
            if n_channels > 1:
                audio_array = audio_array.reshape(-1, n_channels)
                audio_array = np.mean(audio_array, axis=1)

            # Calculate energy in chunks
            chunk_size = sample_rate // frame_rate
            energy = []

            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i : i + chunk_size]
                if len(chunk) > 0:
                    chunk_energy = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                    energy.append(chunk_energy)

            if energy:
                energy = np.array(energy)
                return energy / (np.max(energy) + 1e-6)
            else:
                return np.array([])

    except Exception as e:
        print(f"[AudioAnalysis] Error reading WAV file: {e}")
        return np.array([])


def estimate_mouth_movement(
    energy: np.ndarray,
    threshold: float = 0.3,
    smoothing_window: int = 3
) -> np.ndarray:
    """
    Estimate mouth movement from audio energy.

    Args:
        energy: Energy array from audio
        threshold: Threshold for detecting speech
        smoothing_window: Window size for temporal smoothing

    Returns:
        Mouth movement intensity array (0.0-1.0)
    """
    # Detect speech
    speech_detected = energy > threshold

    # Apply smoothing
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        smooth_energy = np.convolve(energy, kernel, mode="same")
    else:
        smooth_energy = energy

    # Normalize
    if np.max(smooth_energy) > 0:
        movement = smooth_energy / np.max(smooth_energy)
    else:
        movement = smooth_energy

    return np.clip(movement, 0.0, 1.0)


def get_voice_activity(
    audio_path: str,
    frame_rate: int = 30,
    threshold: float = 0.1
) -> Optional[np.ndarray]:
    """
    Detect voice activity in audio.

    Args:
        audio_path: Path to audio file
        frame_rate: Frame rate for alignment
        threshold: Energy threshold for voice detection

    Returns:
        Binary voice activity array, or None if unavailable
    """
    try:
        energy = get_audio_energy(audio_path, frame_rate)
        if energy is None:
            return None

        vad = (energy > threshold).astype(np.float32)
        return vad
    except Exception as e:
        print(f"[AudioAnalysis] Error detecting voice activity: {e}")
        return None


def estimate_phoneme_sequence(
    audio_path: str,
    frame_rate: int = 30
) -> Optional[List[Tuple[str, int, int]]]:
    """
    Estimate phoneme sequence from audio (requires librosa or external tool).

    This is a placeholder for advanced phoneme recognition.
    In production, you would use a speech recognition library.

    Args:
        audio_path: Path to audio file
        frame_rate: Frame rate for alignment

    Returns:
        List of (phoneme, start_frame, end_frame), or None if unavailable
    """
    # Note: This requires external phoneme recognition
    # For now, return None. In production, integrate with:
    # - Google Speech API
    # - Hugging Face wav2vec2
    # - PyAudio + CMU Sphinx
    # - Montreal Forced Aligner
    return None


def predict_mouth_shape_from_audio(
    audio_path: str,
    frame_rate: int = 30
) -> Optional[np.ndarray]:
    """
    Predict mouth shape sequence from audio.

    Args:
        audio_path: Path to audio file
        frame_rate: Frame rate for alignment

    Returns:
        Array of mouth shapes (0=closed, 1=slightly_open, 2=open, 3=very_open)
    """
    try:
        # Try to get phoneme sequence
        phonemes = estimate_phoneme_sequence(audio_path, frame_rate)

        if phonemes is not None:
            # Convert phonemes to mouth shapes
            n_frames = int(get_audio_duration(audio_path) * frame_rate)
            mouth_shapes = np.zeros(n_frames, dtype=np.int32)

            for phoneme, start_frame, end_frame in phonemes:
                shape = PHONEME_TO_MOUTH_SHAPE.get(phoneme, "slightly_open")
                shape_id = ["closed", "slightly_open", "open", "very_open"].index(shape)
                mouth_shapes[start_frame:end_frame] = shape_id

            return mouth_shapes

        else:
            # Fallback to energy-based prediction
            energy = get_audio_energy(audio_path, frame_rate)
            if energy is None:
                return None

            # Simple mapping: energy -> mouth openness
            mouth_shapes = np.zeros(len(energy), dtype=np.int32)
            mouth_shapes[energy < 0.2] = 0  # closed
            mouth_shapes[(energy >= 0.2) & (energy < 0.4)] = 1  # slightly open
            mouth_shapes[(energy >= 0.4) & (energy < 0.7)] = 2  # open
            mouth_shapes[energy >= 0.7] = 3  # very open

            return mouth_shapes

    except Exception as e:
        print(f"[AudioAnalysis] Error predicting mouth shape: {e}")
        return None


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds, or 0.0 if unavailable
    """
    try:
        if has_librosa():
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        else:
            import wave
            with wave.open(audio_path, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                frame_rate = wav_file.getframerate()
                return n_frames / frame_rate
    except Exception as e:
        print(f"[AudioAnalysis] Error getting audio duration: {e}")
        return 0.0


def align_audio_to_frames(
    audio_array: np.ndarray,
    n_frames: int,
    sample_rate: int = 16000,
    frame_rate: int = 30
) -> np.ndarray:
    """
    Align audio array to video frames.

    Args:
        audio_array: Audio samples
        n_frames: Number of video frames
        sample_rate: Audio sample rate
        frame_rate: Video frame rate

    Returns:
        Audio features aligned to frames
    """
    samples_per_frame = sample_rate // frame_rate
    aligned = []

    for i in range(n_frames):
        start = i * samples_per_frame
        end = start + samples_per_frame
        frame_audio = audio_array[start:end]

        if len(frame_audio) > 0:
            energy = np.sqrt(np.mean(frame_audio.astype(np.float32) ** 2))
            aligned.append(energy)
        else:
            aligned.append(0.0)

    return np.array(aligned)
