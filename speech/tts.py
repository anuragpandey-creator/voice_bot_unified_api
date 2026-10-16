"""
speech/tts.py — Unified Text-to-Speech Providers
Supports Smallest AI, Sarvam AI, and Devnagri.
Returns raw audio bytes or base64 audio.
"""

from __future__ import annotations

import base64
import csv
import io
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union
import httpx
from fastapi import HTTPException

from config import cfg
from logger import get_logger
from speech.models import TTSVendor, VoiceGender, SynthesizeResponse

logger = get_logger("speech.tts")

# Language code mapping for Smallest AI
SMALLEST_LANG_CODES: Dict[str, str] = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Bangla": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Malayalam": "ml",
    "Odia": "or",
    "Urdu": "ur",
}

# Sarvam language code mapping
SARVAM_LANG_CODES: Dict[str, str] = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",
    "Bangla": "bn-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Gujarati": "gu-IN",
    "Marathi": "mr-IN",
    "Punjabi": "pa-IN",
    "Odia": "od-IN",
}

_VOICE_MAP: Dict[str, Dict[str, str]] = {}


def _load_voice_map() -> Dict[str, Dict[str, str]]:
    global _VOICE_MAP
    if _VOICE_MAP:
        return _VOICE_MAP

    config_path = Path(cfg.SMALLEST_VOICE_CONFIG_PATH)
    if not config_path.exists():
        logger.warning(f"[VoiceConfig] File not found: {config_path}")
        return {}

    try:
        import openpyxl
        wb = openpyxl.load_workbook(config_path, data_only=True)
        sheet = wb.active
        headers = [str(cell.value or "").strip().lower() for cell in sheet[1]]
        
        name_idx = headers.index("name") if "name" in headers else 0
        lang_idx = headers.index("language") if "language" in headers else 1
        gender_idx = headers.index("gender") if "gender" in headers else 2

        mapping: Dict[str, Dict[str, str]] = {}
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if not row or not row[name_idx]:
                continue
            name = str(row[name_idx]).strip()
            lang = str(row[lang_idx]).strip()
            gender = str(row[gender_idx]).strip().lower()
            if lang not in mapping:
                mapping[lang] = {}
            mapping[lang][gender] = name

        _VOICE_MAP = mapping
        logger.info(f"[VoiceConfig] Loaded voice map with {len(_VOICE_MAP)} languages")
        return _VOICE_MAP
    except Exception as e:
        logger.warning(f"[VoiceConfig] Failed to parse Excel config: {e}")
        return {}


class SmallestProvider:
    """Smallest AI TTS provider."""

    async def synthesize(
        self,
        text: str,
        language: str = "English",
        voice_gender: str = "female",
        sample_rate: int = 24000,
        speed: float = 1.0,
    ) -> bytes:
        voice_map = _load_voice_map()
        lang_key = language.capitalize()
        voice_id = cfg.TTS_VOICE_ID

        if lang_key in voice_map and voice_gender.lower() in voice_map[lang_key]:
            voice_id = voice_map[lang_key][voice_gender.lower()]
        elif voice_gender.lower() == "male":
            voice_id = "magnus"

        payload = {
            "text": text,
            "voice_id": voice_id,
            "sample_rate": sample_rate,
            "speed": speed,
            "add_wav_header": True,
        }

        headers = {
            "Authorization": f"Bearer {cfg.SMALLEST_API_KEY}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(cfg.SMALLEST_TTS_URL, json=payload, headers=headers)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Smallest AI TTS error: {resp.text}")

        return resp.content


class SarvamProvider:
    """Sarvam AI TTS provider."""

    async def synthesize(
        self,
        text: str,
        language: str = "Hindi",
        voice_gender: str = "female",
        sample_rate: int = 24000,
    ) -> bytes:
        lang_code = SARVAM_LANG_CODES.get(language, "hi-IN")
        speaker = "anushka" if voice_gender.lower() == "female" else "arvind"

        payload = {
            "inputs": [text],
            "target_language_code": lang_code,
            "speaker": speaker,
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "speech_sample_rate": sample_rate,
            "enable_preprocessing": True,
            "model": "bulbul:v1",
        }

        headers = {
            "api-subscription-key": cfg.SARVAM_API_KEY,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(cfg.SARVAM_TTS_URL, json=payload, headers=headers)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Sarvam AI TTS error: {resp.text}")

        data = resp.json()
        audios = data.get("audios", [])
        if not audios:
            raise HTTPException(status_code=500, detail="Sarvam returned empty audio list")

        return base64.b64decode(audios[0])


class DevnagriProvider:
    """Devnagri TTS provider."""

    async def synthesize(
        self,
        text: str,
        language: str = "Hindi",
        voice_gender: str = "female",
    ) -> bytes:
        payload = {
            "text": text,
            "language": language.lower(),
            "gender": voice_gender.lower(),
        }
        headers = {"access_token": cfg.DEVNAGRI_API_KEY}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(cfg.DEVNAGRI_TTS_URL, json=payload, headers=headers)

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Devnagri TTS error: {resp.text}")

        data = resp.json()
        audio_b64 = data.get("audio_content") or data.get("audio")
        if audio_b64:
            return base64.b64decode(audio_b64)
        return resp.content


_PROVIDERS = {
    TTSVendor.SMALLEST: SmallestProvider(),
    TTSVendor.SARVAM: SarvamProvider(),
    TTSVendor.DEVNAGRI: DevnagriProvider(),
}


async def synthesize_speech(
    text: str,
    language: str = "Hindi",
    vendor: Union[TTSVendor, str] = TTSVendor.SMALLEST,
    voice_gender: str = "female",
    sample_rate: int = 24000,
) -> SynthesizeResponse:
    """Unified speech synthesis function."""
    t0 = time.perf_counter()
    v_enum = TTSVendor(vendor) if isinstance(vendor, str) else vendor
    provider = _PROVIDERS.get(v_enum, _PROVIDERS[TTSVendor.SMALLEST])

    audio_bytes = await provider.synthesize(
        text=text,
        language=language,
        voice_gender=voice_gender,
        sample_rate=sample_rate,
    )
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

    logger.info(f"[TTS] Synthesized {len(audio_bytes)} bytes | vendor={v_enum.value} lang={language} ms={duration_ms}")
    return SynthesizeResponse(
        audio_base64=b64_audio,
        content_type="audio/wav",
        sample_rate=sample_rate,
        vendor=v_enum.value,
        duration_ms=duration_ms,
    )
