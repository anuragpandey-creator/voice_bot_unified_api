"""
speech/models.py — Data schemas and enumerations for the Speech subsystem
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class STTVendor(str, Enum):
    SMALLEST = "smallest"
    SARVAM = "sarvam"
    DEVNAGRI = "devnagri"


class TTSVendor(str, Enum):
    SMALLEST = "smallest"
    SARVAM = "sarvam"
    DEVNAGRI = "devnagri"


class LIDVendor(str, Enum):
    SARVAM = "sarvam"
    INDIC = "indic"
    GOOGLE = "google"


class VoiceGender(str, Enum):
    FEMALE = "female"
    MALE = "male"


class TranscribeResponse(BaseModel):
    transcript: str
    detected_language: Optional[str] = None
    vendor: str
    duration_ms: float = 0.0


class SynthesizeResponse(BaseModel):
    audio_base64: str
    content_type: str = "audio/wav"
    sample_rate: int = 24000
    vendor: str
    duration_ms: float = 0.0


class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str
    duration_ms: float = 0.0
