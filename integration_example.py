"""
Integration Example: Using Quality and Lip Sync Enhancements

This file demonstrates how to integrate the new quality enhancement and
lip sync features into the Deep-Live-Cam processing pipeline.

Usage:
    python integration_example.py --source face.jpg --target video.mp4 --quality high --enable-audio-sync
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import modules.globals
from modules.processors.frame.quality_enhancer import (
    apply_quality_profile,
    upscale_frame,
    enhance_frame_detail,
)
from modules.processors.frame.lip_sync_enhancer import (
    MouthTracker,
    get_mouth_region,
    blend_mouth_region,
    enhance_mouth_details,
    calculate_mouth_openness,
)
from modules.processors.frame.audio_analysis import (
    extract_audio_from_video,
    get_audio_energy,
    predict_mouth_shape_from_audio,
)


class EnhancedDeepLiveCamProcessor:
    """
    Example processor integrating quality and lip sync enhancements.
    """

    def __init__(
        self,
        quality_mode: str = "high",
        enable_lip_sync: bool = True,
        enable_audio_sync: bool = False,
    ):
        """
        Initialize processor with enhancement settings.

        Args:
            quality_mode: "standard", "high", or "ultra"
            enable_lip_sync: Enable lip sync enhancements
            enable_audio_sync: Enable audio-based lip sync
        """
        self.quality_mode = quality_mode
        self.enable_lip_sync = enable_lip_sync
        self.enable_audio_sync = enable_audio_sync

        # Initialize mouth tracker for temporal smoothing
        self.mouth_tracker = None
        if enable_lip_sync:
            self.mouth_tracker = MouthTracker(history_size=5)

        # Audio sync data
        self.audio_energy = None
        self.mouth_shapes = None
        self.current_frame_idx = 0

    def process_frame(self, frame, faces):
        """
        Process frame with quality and lip sync enhancements.

        Args:
            frame: Input frame (numpy array)
            faces: List of detected faces with landmarks

        Returns:
            Enhanced frame
        """
        result = frame.copy()

        # 1. Apply quality enhancements
        result = self._apply_quality_enhancement(result)

        # 2. Apply lip sync enhancements
        if self.enable_lip_sync and faces:
            result = self._apply_lip_sync_enhancement(result, faces)

        # 3. Apply audio-based sync if available
        if self.enable_audio_sync and self.mouth_shapes is not None:
            result = self._apply_audio_sync(result, faces)

        self.current_frame_idx += 1
        return result

    def _apply_quality_enhancement(self, frame):
        """
        Apply quality profile enhancements.

        Standard: 720p with minimal processing
        High: 1080p with unsharp masking
        Ultra: 1440p+ with full enhancement
        """
        # Upscale if source is lower than target
        if self.quality_mode in ["high", "ultra"]:
            preset = {"high": 1.25, "ultra": 1.5}[self.quality_mode]
            h, w = frame.shape[:2]
            if h < 1080:
                frame = upscale_frame(frame, factor=preset, method="lanczos")

        # Apply quality profile
        frame = apply_quality_profile(frame, self.quality_mode)

        # Detail enhancement for high quality
        if self.quality_mode in ["high", "ultra"]:
            strength = {"high": 0.7, "ultra": 0.9}[self.quality_mode]
            frame = enhance_frame_detail(frame, strength)

        return frame

    def _apply_lip_sync_enhancement(self, frame, faces):
        """
        Apply lip sync enhancements using mouth tracking and blending.
        """
        for face in faces:
            # Skip if no landmarks
            if not hasattr(face, "landmark_2d_106") or face.landmark_2d_106 is None:
                continue

            # Get mouth region
            mouth_region, bbox = get_mouth_region(face, frame, expansion=1.2)
            if mouth_region is None:
                continue

            # Extract mouth landmarks for tracking
            mouth_landmarks = face.landmark_2d_106[48:68]

            # Update mouth tracker for smoothing
            if self.mouth_tracker is not None:
                self.mouth_tracker.update(mouth_landmarks)
                mouth_landmarks = self.mouth_tracker.get_smoothed_landmarks()

            # Calculate mouth openness
            openness = calculate_mouth_openness(mouth_landmarks)

            # Enhance mouth details
            enhanced_mouth = enhance_mouth_details(
                mouth_region,
                brightness_factor=1.1,
                contrast_factor=1.2,
            )

            # Blend mouth region with feathering
            blend_strength = 0.7 if openness > 0.3 else 0.8
            frame = blend_mouth_region(
                frame,
                enhanced_mouth,
                mouth_region,
                bbox,
                blend_strength=blend_strength,
                feather_size=25,
            )

        return frame

    def _apply_audio_sync(self, frame, faces):
        """
        Apply audio-based mouth shape modifications.
        """
        if self.current_frame_idx >= len(self.mouth_shapes):
            return frame

        predicted_shape = self.mouth_shapes[self.current_frame_idx]

        for face in faces:
            # Get mouth region
            mouth_region, bbox = get_mouth_region(face, frame, expansion=1.2)
            if mouth_region is None:
                continue

            # Modify mouth region based on predicted shape
            # 0=closed, 1=slightly_open, 2=open, 3=very_open
            if predicted_shape == 0:
                # Mouth closed - reduce vertical dimension
                modification = self._modify_mouth_closed(mouth_region)
            elif predicted_shape == 1:
                # Slightly open
                modification = self._modify_mouth_slightly_open(mouth_region)
            elif predicted_shape == 2:
                # Open
                modification = self._modify_mouth_open(mouth_region)
            else:
                # Very open
                modification = self._modify_mouth_very_open(mouth_region)

            # Blend back
            frame = blend_mouth_region(
                frame,
                modification,
                mouth_region,
                bbox,
                blend_strength=0.6,
                feather_size=20,
            )

        return frame

    @staticmethod
    def _modify_mouth_closed(mouth_region):
        """Compress mouth region vertically for closed mouth."""
        h, w = mouth_region.shape[:2]
        center_y = h // 2
        compression = 0.7

        # Create compressed version
        compressed = np.zeros_like(mouth_region)
        new_h = int(h * compression)
        offset = (h - new_h) // 2

        resized = cv2.resize(
            mouth_region, (w, new_h), interpolation=cv2.INTER_CUBIC
        )
        compressed[offset : offset + new_h, :] = resized

        # Blend with original
        return cv2.addWeighted(mouth_region, 0.3, compressed, 0.7, 0)

    @staticmethod
    def _modify_mouth_slightly_open(mouth_region):
        """Minimal modification for slightly open mouth."""
        return mouth_region.copy()

    @staticmethod
    def _modify_mouth_open(mouth_region):
        """Slightly expand mouth region for open mouth."""
        h, w = mouth_region.shape[:2]
        expanded = cv2.resize(
            mouth_region, (int(w * 1.05), int(h * 1.1)), interpolation=cv2.INTER_CUBIC
        )

        # Center the expanded region
        result = np.zeros_like(mouth_region)
        y_offset = (mouth_region.shape[0] - expanded.shape[0]) // 2
        x_offset = (mouth_region.shape[1] - expanded.shape[1]) // 2

        y1 = max(0, y_offset)
        y2 = min(mouth_region.shape[0], y_offset + expanded.shape[0])
        x1 = max(0, x_offset)
        x2 = min(mouth_region.shape[1], x_offset + expanded.shape[1])

        result[y1:y2, x1:x2] = expanded[
            max(0, -y_offset) : max(0, -y_offset) + (y2 - y1),
            max(0, -x_offset) : max(0, -x_offset) + (x2 - x1),
        ]

        return result

    @staticmethod
    def _modify_mouth_very_open(mouth_region):
        """Expand mouth region for very open mouth."""
        h, w = mouth_region.shape[:2]
        expanded = cv2.resize(
            mouth_region, (int(w * 1.1), int(h * 1.2)), interpolation=cv2.INTER_CUBIC
        )

        result = np.zeros_like(mouth_region)
        y_offset = (mouth_region.shape[0] - expanded.shape[0]) // 2
        x_offset = (mouth_region.shape[1] - expanded.shape[1]) // 2

        y1 = max(0, y_offset)
        y2 = min(mouth_region.shape[0], y_offset + expanded.shape[0])
        x1 = max(0, x_offset)
        x2 = min(mouth_region.shape[1], x_offset + expanded.shape[1])

        result[y1:y2, x1:x2] = expanded[
            max(0, -y_offset) : max(0, -y_offset) + (y2 - y1),
            max(0, -x_offset) : max(0, -x_offset) + (x2 - x1),
        ]

        return result

    def setup_audio_sync(self, video_path):
        """
        Setup audio-based lip sync for video.

        Args:
            video_path: Path to input video
        """
        if not self.enable_audio_sync:
            return False

        print("[Integration] Extracting audio...")
        audio_path = "temp_audio.wav"
        if not extract_audio_from_video(video_path, audio_path):
            print("[Integration] Failed to extract audio")
            return False

        print("[Integration] Analyzing audio...")
        self.audio_energy = get_audio_energy(audio_path, frame_rate=30)
        self.mouth_shapes = predict_mouth_shape_from_audio(audio_path, frame_rate=30)

        if self.mouth_shapes is None:
            print("[Integration] Failed to predict mouth shapes")
            return False

        print(f"[Integration] Loaded audio sync for {len(self.mouth_shapes)} frames")
        return True


def main():
    """
    Example usage of EnhancedDeepLiveCamProcessor.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", default="high", choices=["standard", "high", "ultra"])
    parser.add_argument("--enable-lip-sync", action="store_true", default=True)
    parser.add_argument("--enable-audio-sync", action="store_true", default=False)
    args = parser.parse_args()

    # Initialize processor
    processor = EnhancedDeepLiveCamProcessor(
        quality_mode=args.quality,
        enable_lip_sync=args.enable_lip_sync,
        enable_audio_sync=args.enable_audio_sync,
    )

    print(f"[Integration] Initialized processor with quality_mode={args.quality}")
    print(f"[Integration] Lip sync enabled: {args.enable_lip_sync}")
    print(f"[Integration] Audio sync enabled: {args.enable_audio_sync}")

    # Example frame processing
    # In real usage, this would be integrated into the main deepfake pipeline
    example_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    example_frame[:] = (100, 100, 100)  # Gray background

    # Process frame
    processed = processor.process_frame(example_frame, faces=[])
    print(f"[Integration] Processed frame shape: {processed.shape}")


if __name__ == "__main__":
    main()
