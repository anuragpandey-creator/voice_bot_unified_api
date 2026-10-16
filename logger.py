"""
logger.py — Central Loguru-based logger setup for Voice Bot Core
"""

import os
import sys
from loguru import logger as _loguru_logger

from config import cfg

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

_loguru_logger.remove()

# Console Sink
_loguru_logger.add(
    sys.stdout,
    level=cfg.LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}",
)

# File Sink (logs/voice_bot.log, rotates at 20MB, keeps last 30 days)
_loguru_logger.add(
    "logs/voice_bot.log",
    level=cfg.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    rotation="20 MB",
    retention="30 days",
    enqueue=True,
)


def get_logger(name: str = __name__):
    return _loguru_logger.bind(name=name)
