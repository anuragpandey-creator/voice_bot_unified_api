"""
pipeline/voice_pipeline.py — End-to-End Voice Bot Pipeline Engine
Connects:
Audio In -> STT -> Language Detection -> Translation -> Orchestrator (TTT) -> Translation -> TTS Audio Stream
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Union
from fastapi import UploadFile

from config import cfg
from logger import get_logger
from speech.stt import transcribe_audio
from speech.tts import synthesize_speech
from speech.lid import detect_language
from speech.translation import native_to_english, english_to_native
from ttt_engine.orchestrator import _process_message_stream, start_chat, StartChatRequest
from ttt_engine.memory_store import memory_store

logger = get_logger("pipeline.voice")

# Punctuation marks for sentence boundary segmentation
SENTENCE_PUNCTUATION = (".", "!", "?", "।", "؟", "\n")
MAX_FRAGMENT_CHARS = 160


class VoicePipeline:
    """Orchestrates end-to-end multi-turn voice interaction."""

    async def start_call(
        self,
        agent_id: str,
        customer_id: Optional[str] = None,
        tts_vendor: str = "smallest",
        voice_gender: str = "female",
    ) -> Dict[str, Any]:
        """Initiates a new voice call, loads CBS data, and synthesizes the opening greeting."""
        t0 = time.perf_counter()
        req = StartChatRequest(agent_id=agent_id, customer_id=customer_id)
        start_res = await start_chat(req)

        greeting_text = start_res.greeting
        session_id = start_res.session_id

        # Synthesize opening greeting
        tts_res = await synthesize_speech(
            text=greeting_text,
            language="English",
            vendor=tts_vendor,
            voice_gender=voice_gender,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(f"[VoiceCall] Call started | session_id={session_id} agent_id={agent_id} ms={elapsed_ms}")

        return {
            "session_id": session_id,
            "greeting_text": greeting_text,
            "greeting_audio_base64": tts_res.audio_base64,
            "current_node_id": getattr(start_res, "current_node", getattr(start_res, "current_node_id", "")),
            "current_node_name": getattr(start_res, "current_node_name", ""),
            "variables": getattr(start_res, "variables", {}),
            "duration_ms": elapsed_ms,
        }

    async def process_voice_turn(
        self,
        audio_input: Union[UploadFile, str, bytes],
        session_id: str,
        stt_vendor: str = "smallest",
        tts_vendor: str = "smallest",
        voice_gender: str = "female",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes a single user voice turn:
        1. Transcribes audio to text via STT.
        2. Detects native language and translates to English if necessary.
        3. Drives the state-machine orchestrator token/sentence stream.
        4. Synthesizes each generated sentence via TTS in real-time.
        Yields events as they happen for low-latency streaming to the client.
        """
        turn_start = time.perf_counter()

        # Step 1: Speech-to-Text
        stt_t0 = time.perf_counter()
        stt_res = await transcribe_audio(audio_input, vendor=stt_vendor)
        user_transcript = stt_res.transcript.strip()
        stt_ms = round((time.perf_counter() - stt_t0) * 1000, 2)

        if not user_transcript:
            yield {
                "type": "error",
                "content": "No audible speech detected in input.",
                "duration_ms": stt_ms,
            }
            return

        # Step 2: Language Identification
        user_lang = stt_res.detected_language or await detect_language(user_transcript)
        logger.info(f"[VoiceTurn] STT complete: '{user_transcript}' | lang={user_lang} | stt_ms={stt_ms}")

        # Step 3: Translate to English (Orchestrator runs in English)
        english_text = user_transcript
        if user_lang.lower() not in ("english", "en"):
            english_text = await native_to_english(user_transcript, source_language=user_lang)
            logger.info(f"[VoiceTurn] Translated to EN: '{english_text}'")

        # Yield transcript info to client immediately
        yield {
            "type": "transcript",
            "transcript": user_transcript,
            "english_text": english_text,
            "language": user_lang,
            "stt_duration_ms": stt_ms,
        }

        # Step 4: Stream through the TTT Orchestrator Engine
        sentence_buffer = []
        sentence_count = 0

        async for event in _process_message_stream(session_id, english_text):
            event_type = event.get("type")

            if event_type == "token":
                token = event.get("content", "")
                sentence_buffer.append(token)
                accumulated = "".join(sentence_buffer)

                # Check if sentence is complete
                if any(accumulated.rstrip().endswith(p) for p in SENTENCE_PUNCTUATION) or len(accumulated) >= MAX_FRAGMENT_CHARS:
                    sentence_text = accumulated.strip()
                    if sentence_text:
                        sentence_count += 1
                        sentence_t0 = time.perf_counter()

                        # Translate sentence back to native language if user spoke in native
                        native_sentence = sentence_text
                        if user_lang.lower() not in ("english", "en"):
                            native_sentence = await english_to_native(sentence_text, target_language=user_lang)

                        # Synthesize speech chunk
                        tts_res = await synthesize_speech(
                            text=native_sentence,
                            language=user_lang,
                            vendor=tts_vendor,
                            voice_gender=voice_gender,
                        )
                        sentence_ms = round((time.perf_counter() - sentence_t0) * 1000, 2)

                        yield {
                            "type": "sentence_audio",
                            "sentence_index": sentence_count,
                            "english_sentence": sentence_text,
                            "native_sentence": native_sentence,
                            "audio_base64": tts_res.audio_base64,
                            "tts_duration_ms": sentence_ms,
                        }

                    sentence_buffer = []

            elif event_type == "done":
                # Flush any remaining tokens in buffer
                if sentence_buffer:
                    sentence_text = "".join(sentence_buffer).strip()
                    if sentence_text:
                        sentence_count += 1
                        native_sentence = sentence_text
                        if user_lang.lower() not in ("english", "en"):
                            native_sentence = await english_to_native(sentence_text, target_language=user_lang)

                        tts_res = await synthesize_speech(
                            text=native_sentence,
                            language=user_lang,
                            vendor=tts_vendor,
                            voice_gender=voice_gender,
                        )
                        yield {
                            "type": "sentence_audio",
                            "sentence_index": sentence_count,
                            "english_sentence": sentence_text,
                            "native_sentence": native_sentence,
                            "audio_base64": tts_res.audio_base64,
                        }
                    sentence_buffer = []

                # Yield turn completion event
                total_turn_ms = round((time.perf_counter() - turn_start) * 1000, 2)
                yield {
                    "type": "turn_complete",
                    "full_response": event.get("response", ""),
                    "current_node_name": event.get("current_node_name", ""),
                    "transitioned": event.get("transitioned", False),
                    "total_turn_duration_ms": total_turn_ms,
                }

            elif event_type == "error":
                yield event


voice_pipeline = VoicePipeline()
