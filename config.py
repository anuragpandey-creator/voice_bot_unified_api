"""
config.py — Central unified configuration for Voice Bot Core
Loads all settings from root .env with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
_ENV_PATH = _ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()


class Config:
    # ── Server ──────────────────────────────────────────────────────────────
    PORT: int = int(os.getenv("PORT", "1224"))
    GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7870"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Database & Storage ──────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://kftadmin:Kftadmin%402026@localhost:5432/unified_configuration",
    )
    USE_DB: bool = os.getenv("USE_DB", "true").lower() == "true"
    LOCAL_AGENT_CONFIG_PATH: str = os.getenv(
        "LOCAL_AGENT_CONFIG_PATH",
        str(_ROOT / "data" / "agent_config.json"),
    )
    DATA_DIR: str = os.getenv("DATA_DIR", str(_ROOT / "data"))
    CBS_DATA_DIR: str = os.getenv(
        "CBS_DATA_DIR",
        str(_ROOT / "data" / "customer_master_data"),
    )
    SESSION_CACHE_TTL_SEC: int = int(os.getenv("SESSION_CACHE_TTL_SEC", "3600"))
    ERROR_LOG_RETENTION_DAYS: int = int(os.getenv("ERROR_LOG_RETENTION_DAYS", "90"))

    # ── LLM (TTT Engine) ────────────────────────────────────────────────────
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        r"E:\DS_Team_Share\hugging-face-model\models--Qwen--Qwen2.5-7B-Instruct-1M\snapshots\e28526f7bb80e2a9c8af03b831a9af3812f18fba",
    )
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "60"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    TOP_K: int = int(os.getenv("TOP_K", "40"))

    FLEX_MODE_ENABLED: bool = os.getenv("FLEX_MODE_ENABLED", "false").lower() == "true"
    LENDER_NAME: str = os.getenv("LENDER_NAME", "Knight Gini")

    # KV-Cache settings
    KV_CACHE_ENABLED: bool = os.getenv("KV_CACHE_ENABLED", "true").lower() == "true"
    KV_CACHE_MAX_ENTRIES: int = int(os.getenv("KV_CACHE_MAX_ENTRIES", "4"))

    # Concurrency queue
    MAX_CONCURRENT_LLM: int = int(os.getenv("MAX_CONCURRENT_LLM", "1"))
    LLM_QUEUE_SIZE: int = int(os.getenv("LLM_QUEUE_SIZE", "50"))
    LLM_REQUEST_TIMEOUT_SEC: int = int(os.getenv("LLM_REQUEST_TIMEOUT_SEC", "60"))

    # ── Smallest AI (STT & TTS) ─────────────────────────────────────────────
    SMALLEST_API_KEY: str = os.getenv("SMALLEST_API_KEY", "sk_eb0e341706583c82430bceb6f1221451")
    SMALLEST_STT_URL: str = os.getenv("SMALLEST_STT_URL", "https://api.smallest.ai/waves/v1/pulse/get_text")
    SMALLEST_TTS_URL: str = os.getenv("SMALLEST_TTS_URL", "https://api.smallest.ai/waves/v1/lightning-v3.1/get_speech")
    SMALLEST_VOICE_CONFIG_PATH: str = os.getenv(
        "SMALLEST_VOICE_CONFIG_PATH",
        str(_ROOT / "data" / "voices_configuration.xlsx"),
    )
    TTS_VOICE_ID: str = os.getenv("TTS_VOICE_ID", "kavya")

    # ── Sarvam AI (STT, TTS, LID, Translation) ──────────────────────────────
    SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "sk_655xhu78_4X4UICikstyFsdJL52qcz2Hm")
    SARVAM_STT_URL: str = os.getenv("SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text")
    SARVAM_TTS_URL: str = os.getenv("SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech")
    SARVAM_LID_URL: str = os.getenv("SARVAM_LID_URL", "http://192.168.0.155:1230/sarvamapi-language-detection")

    # ── Devnagri (STT & TTS) ────────────────────────────────────────────────
    DEVNAGRI_API_KEY: str = os.getenv("DEVNAGRI_API_KEY", "e91f7f69ca8fec6acf6bcdcca91428c47870dbdc8c248d040967848357e1ea95")
    DEVNAGRI_STT_URL: str = os.getenv("DEVNAGRI_STT_URL", "https://audio.devnagri.com/demo/gtm-transcribe")
    DEVNAGRI_TTS_URL: str = os.getenv("DEVNAGRI_TTS_URL", "https://audio.devnagri.com/demo/text_to_speech")

    # ── Translation Endpoints ───────────────────────────────────────────────
    NATIVE_TO_ENGLISH_URL: str = os.getenv("NATIVE_TO_ENGLISH_URL", "http://192.168.0.155:1206/native-to-english")
    ENGLISH_TO_NATIVE_URL: str = os.getenv("ENGLISH_TO_NATIVE_URL", "http://192.168.0.155:1206/english-to-native")


cfg = Config()
