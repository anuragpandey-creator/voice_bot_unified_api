"""
speech/stt.py — Unified Speech-to-Text Providers
Supports Smallest AI, Sarvam AI, and Devnagri.
Accepts audio as UploadFile, file path (str), or raw bytes.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any, Optional, Tuple, Union
import httpx
from fastapi import HTTPException, UploadFile

from config import cfg
from logger import get_logger
from speech.models import STTVendor, TranscribeResponse

logger = get_logger("speech.stt")


async def _extract_audio_bytes(
    audio: Union[UploadFile, str, bytes, Any],
) -> Tuple[bytes, str, str]:
    """Helper to extract raw bytes, filename, and content_type from various input types."""
    if hasattr(audio, "read"):
        read_fn = audio.read
        if asyncio.iscoroutinefunction(read_fn):
            data = await read_fn()
        else:
            data = read_fn()
        if hasattr(audio, "seek"):
            seek_fn = audio.seek
            if asyncio.iscoroutinefunction(seek_fn):
                await seek_fn(0)
            else:
                seek_fn(0)
        filename = getattr(audio, "filename", None) or "audio.wav"
        content_type = getattr(audio, "content_type", None) or "audio/wav"
        return data, filename, content_type
    elif isinstance(audio, (bytes, bytearray)):
        return bytes(audio), "audio.wav", "audio/wav"
    elif isinstance(audio, str):
        with open(audio, "rb") as f:
            data = f.read()
        filename = audio.split("/")[-1].split("\\")[-1]
        content_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg"
        return data, filename, content_type
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio)}")


class SmallestSTTProvider:
    """Smallest AI Speech-to-Text."""

    async def transcribe(self, audio: Union[UploadFile, str, bytes]) -> Tuple[str, Optional[str]]:
        audio_bytes, filename, content_type = await _extract_audio_bytes(audio)
        headers = {
            "Authorization": f"Bearer {cfg.SMALLEST_API_KEY}",
            "Content-Type": content_type,
        }
        logger.info(f"[STT][Smallest] Transcribing {len(audio_bytes)} bytes")

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            http2=True,
        ) as client:
            response = await client.post(
                f"{cfg.SMALLEST_STT_URL}?language=multi-indic",
                headers=headers,
                content=audio_bytes,
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Smallest AI STT error: {response.text}")

        data = response.json()
        transcript = data.get("transcription") or data.get("text") or data.get("transcript") or str(data)
        return transcript.strip(), None


class SarvamSTTProvider:
    """Sarvam AI Speech-to-Text (saaras:v3)."""

    async def transcribe(self, audio: Union[UploadFile, str, bytes]) -> Tuple[str, Optional[str]]:
        audio_bytes, filename, content_type = await _extract_audio_bytes(audio)
        headers = {
            "api-subscription-key": cfg.SARVAM_API_KEY,
        }
        files = {
            "file": (filename, audio_bytes, content_type),
        }
        data = {
            "model": "saaras:v3",
            "language_code": "unknown",
        }
        logger.info(f"[STT][Sarvam] Transcribing {len(audio_bytes)} bytes")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                cfg.SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Sarvam AI STT error: {response.text}")

        resp_data = response.json()
        transcript = (
            resp_data.get("transcript")
            or resp_data.get("text")
            or resp_data.get("transcription")
            or str(resp_data)
        )
        detected_language = resp_data.get("language_code") or resp_data.get("language")
        return transcript.strip(), detected_language


class DevnagriSTTProvider:
    """Devnagri Speech-to-Text."""

    async def transcribe(self, audio: Union[UploadFile, str, bytes]) -> Tuple[str, Optional[str]]:
        audio_bytes, filename, content_type = await _extract_audio_bytes(audio)
        headers = {"access_token": cfg.DEVNAGRI_API_KEY}
        files = {"audio_file": (filename, audio_bytes, content_type)}
        logger.info(f"[STT][Devnagri] Transcribing {len(audio_bytes)} bytes")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                cfg.DEVNAGRI_STT_URL,
                headers=headers,
                files=files,
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Devnagri STT error: {response.text}")

        data = response.json()
        transcript = data.get("transcript") or data.get("text") or data.get("result") or str(data)
        detected_language = data.get("language") or data.get("language_code")
        return transcript.strip(), detected_language


_PROVIDERS = {
    STTVendor.SMALLEST: SmallestSTTProvider(),
    STTVendor.SARVAM: SarvamSTTProvider(),
    STTVendor.DEVNAGRI: DevnagriSTTProvider(),
}


async def transcribe_audio(
    audio: Union[UploadFile, str, bytes],
    vendor: Union[STTVendor, str] = STTVendor.SMALLEST,
) -> TranscribeResponse:
    """Unified transcribe function."""
    t0 = time.perf_counter()
    v_enum = STTVendor(vendor) if isinstance(vendor, str) else vendor
    provider = _PROVIDERS.get(v_enum, _PROVIDERS[STTVendor.SMALLEST])

    transcript, detected_lang = await provider.transcribe(audio)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(f"[STT] Completed | vendor={v_enum.value} lang={detected_lang} ms={duration_ms}")
    return TranscribeResponse(
        transcript=transcript,
        detected_language=detected_lang,
        vendor=v_enum.value,
        duration_ms=duration_ms,
    )
