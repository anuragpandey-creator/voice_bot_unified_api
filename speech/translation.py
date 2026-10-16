"""
speech/translation.py — Bidirectional Translation Services
Translates between Indian regional languages and English.
Uses Sarvam AI Translate (API / SDK) or external translation endpoints.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
import httpx

from config import cfg
from logger import get_logger
from speech.models import TranslationResponse

logger = get_logger("speech.translation")

SARVAM_LANG_CODES = {
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Bangla": "bn-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Kannada": "kn-IN",
    "Gujarati": "gu-IN",
    "Punjabi": "pa-IN",
    "Malayalam": "ml-IN",
    "Odia": "or-IN",
    "English": "en-IN",
}


async def native_to_english(text: str, source_language: str = "Hindi") -> str:
    """Translates native text into English."""
    if not text or not text.strip():
        return ""
    if source_language.lower() in ("english", "en"):
        return text

    t0 = time.perf_counter()
    # 1. Try Sarvam AI SDK
    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=cfg.SARVAM_API_KEY)
        src_code = SARVAM_LANG_CODES.get(source_language, "hi-IN")
        
        def _call_sdk():
            resp = client.text.translate(
                input=text,
                source_language_code=src_code,
                target_language_code="en-IN",
                mode="classic-colloquial",
                model="mayura:v1",
            )
            return getattr(resp, "translated_text", None) or resp.get("translated_text", "")

        translated = await asyncio.to_thread(_call_sdk)
        if translated:
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(f"[Translate] {source_language} -> EN took {elapsed}ms")
            return translated
    except Exception as exc:
        logger.debug(f"[Translate] Sarvam SDK translate failed: {exc}")

    # 2. Try HTTP endpoint fallback
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(
                cfg.NATIVE_TO_ENGLISH_URL,
                json={"text": text, "actual_language": source_language, "provider": "sarvam_web"},
            )
            if res.status_code == 200:
                data = res.json()
                return data.get("english_text") or data.get("translated_text") or text
    except Exception as exc:
        logger.warning(f"[Translate] Native->EN HTTP fallback failed: {exc}")

    return text


async def english_to_native(text: str, target_language: str = "Hindi") -> str:
    """Translates English text into target native script."""
    if not text or not text.strip():
        return ""
    if target_language.lower() in ("english", "en"):
        return text

    t0 = time.perf_counter()
    # 1. Try Sarvam AI SDK
    try:
        from sarvamai import SarvamAI
        client = SarvamAI(api_subscription_key=cfg.SARVAM_API_KEY)
        tgt_code = SARVAM_LANG_CODES.get(target_language, "hi-IN")
        
        def _call_sdk():
            resp = client.text.translate(
                input=text,
                source_language_code="en-IN",
                target_language_code=tgt_code,
                mode="classic-colloquial",
                model="mayura:v1",
                output_script="fully-native",
            )
            return getattr(resp, "translated_text", None) or resp.get("translated_text", "")

        translated = await asyncio.to_thread(_call_sdk)
        if translated:
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(f"[Translate] EN -> {target_language} took {elapsed}ms")
            return translated
    except Exception as exc:
        logger.debug(f"[Translate] Sarvam SDK translate failed: {exc}")

    # 2. Try HTTP endpoint fallback
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(
                cfg.ENGLISH_TO_NATIVE_URL,
                json={"text": text, "target_language": target_language, "provider": "sarvam_web"},
            )
            if res.status_code == 200:
                data = res.json()
                return data.get("translated_text") or data.get("native_text") or text
    except Exception as exc:
        logger.warning(f"[Translate] EN->Native HTTP fallback failed: {exc}")

    return text
