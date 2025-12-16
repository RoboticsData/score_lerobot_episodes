"""
score_lerobot_episodes - Utilities for scoring LeRobot episodes

This package provides tools for:
- Loading and processing LeRobot datasets
- Scoring video quality (blur, darkness, contrast)
- Evaluating episode quality
"""

# VLM interface (optional)
try:
    from .vlm import VLMInterface
except ImportError:
    VLMInterface = None


__version__ = "0.1.0"

