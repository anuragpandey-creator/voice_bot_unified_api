"""
speech/lid.py — Language Identification (LID) Providers
Supports Sarvam AI (SDK / HTTP) and Google/Mock LID.
"""

from __future__ import annotations

import asyncio
from typing import Optional
import httpx

from config import cfg
from logger import get_logger

logger = get_logger("speech.lid")

SARVAM_CODE_TO_LANG = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
}


async def detect_language(text: str, vendor: str = "sarvam") -> str:
    """Detects spoken language of given text string."""
    if not text or not text.strip():
        return "English"

    if vendor.lower() == "sarvam":
        try:
            # 1. Try SDK if available
            try:
                from sarvamai import SarvamAI
                client = SarvamAI(api_subscription_key=cfg.SARVAM_API_KEY)
                resp = await asyncio.to_thread(client.text.identify_language, input=text)
                lang_code = resp.language_code
                return SARVAM_CODE_TO_LANG.get(lang_code, "Hindi")
            except Exception as sdk_err:
                logger.debug(f"[LID] Sarvam SDK call failed, trying HTTP: {sdk_err}")

            # 2. Try HTTP endpoint
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(cfg.SARVAM_LID_URL, json={"text": text})
                if res.status_code == 200:
                    data = res.json()
                    return data.get("language") or data.get("language_code") or "Hindi"
        except Exception as e:
            logger.warning(f"[LID] Sarvam detection fallback: {e}")

    # Fallback default
    return "Hindi" if any(ord(c) > 127 for c in text) else "English"
