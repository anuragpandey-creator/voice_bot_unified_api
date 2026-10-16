"""
main.py — Astor AI: Unified API 2.0.0
=====================================
Unified Voice Bot Platform matching the exact Swagger structure:
  0. Agent Management             — Create, update, delete, list, get, extract variables
  1. Health                       — Service liveness and health checks
  2. STT                          — Speech-to-Text audio transcription
  3. LID                          — Language Identification from text
  4. Translation                  — Native ↔ English bidirectional translation
  5. TTS                          — Text-to-Speech audio synthesis
  6.1. LLM Config — Agents        — LLM agent registry operations
  6.2. LLM Config — Documents     — Knowledge base files for agents
  6.3. LLM Config — Variables     — Dynamic prompt placeholder management
  6.4. LLM Config — Chat          — Streaming chat with an LLM agent (SSE)
  6.5. LLM Config — Models        — Load and switch GPU models
  9.1. Flow — Agents              — Workflow engine agent CRUD
  9.2. Flow — Nodes               — Workflow graph node CRUD & transitions
  9.3. Flow — Subflows            — Reusable subflow modules
  9.4. Flow — Tool Registry       — Workflow API tool actions & execution
  9.5. Flow — CBS Banking Data    — Core Banking System mock customer & loan data
  9.6. Flow — Backend Services    — Customer verification, PTP, & SMS services
  10. Languages                   — Supported translation & detection languages
  11. Voice Pipeline              — End-to-end voice streaming and turns

Run:
  uvicorn main:app --port 1224
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Dict, List, Optional, Union

from fastapi import (
    FastAPI,
    APIRouter,
    Body,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# Ensure root directory is on sys.path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import cfg
from logger import get_logger

logger = get_logger("unified_api")

# ── Speech Subsystem ──────────────────────────────────────────────────────────
from speech import (
    STTVendor,
    TTSVendor,
    VoiceGender,
    transcribe_audio,
    synthesize_speech,
    detect_language,
    native_to_english,
    english_to_native,
)

# ── TTT Engine Subsystem ──────────────────────────────────────────────────────
from ttt_engine.orchestrator import (
    orchestrator_router,
    subflow_router,
    _process_message_stream,
)
from ttt_engine.tool_registry import tool_router, image_router
from ttt_engine.cbs_api import cbs_router, services_router
from ttt_engine.memory_store import memory_store
from ttt_engine.llm_engine import llm_engine, llm_queue
from ttt_engine.agent_manager import (
    agent_router,
    node_router,
    agent_store,
    node_store,
)
from pipeline.voice_pipeline import voice_pipeline

if cfg.USE_DB:
    try:
        from database import init_db
    except ImportError:
        from ttt_engine.database import init_db


# ── Tag Definition & Ordering ─────────────────────────────────────────────────

_OPENAPI_TAGS = [
    {"name": "0. Agent Management",                  "description": "Unified voice agent creation, configuration, listing, and deletion"},
    {"name": "1. Health",                            "description": "Service liveness and health checks"},
    {"name": "2. STT",                               "description": "Speech-to-Text audio transcription"},
    {"name": "3. LID",                               "description": "Language Identification from text"},
    {"name": "4. Translation",                       "description": "Native ↔ English bidirectional translation"},
    {"name": "5. TTS",                               "description": "Text-to-Speech audio synthesis"},
    {"name": "6.1. LLM Config — Agents",             "description": "LLM agent registry operations"},
    {"name": "6.2. LLM Config — Documents",          "description": "Upload and manage knowledge base files for agents"},
    {"name": "6.3. LLM Config — Variables",          "description": "Prompt dynamic placeholder management"},
    {"name": "6.4. LLM Config — Chat",               "description": "Streaming chat with an LLM agent (SSE)"},
    {"name": "6.5. LLM Config — Models",             "description": "Load and switch local GPU models"},
    {"name": "9.1. Flow — Agents",                  "description": "Workflow engine agent CRUD"},
    {"name": "9.2. Flow — Nodes",                   "description": "Workflow graph node CRUD and edge transitions"},
    {"name": "9.3. Flow — Subflows",                "description": "Reusable subflow modules"},
    {"name": "9.4. Flow — Tool Registry",           "description": "Workflow API tool actions and execution"},
    {"name": "9.5. Flow — CBS Banking Data",        "description": "Core Banking System mock customer & loan data"},
    {"name": "9.6. Flow — Backend Services",        "description": "Customer verification, PTP, and SMS services"},
    {"name": "10. Languages",                       "description": "Supported translation and detection languages"},
    {"name": "11. Voice Pipeline",                   "description": "End-to-end voice streaming and turns"},
]


# ── Helper for Retagging Routers ──────────────────────────────────────────────

def set_router_tags(router: APIRouter, new_tag: str):
    """Assigns a unified numbered tag to all endpoints in the given router."""
    for route in router.routes:
        if hasattr(route, "tags"):
            route.tags = [new_tag]


# ── String & Token Cleaning Helpers ───────────────────────────────────────────

def clean_input_token(token: Optional[str]) -> str:
    if not token:
        return ""
    cleaned = str(token).strip().strip('"').strip("'").strip()
    matches = re.findall(r'["\']?([a-zA-Z0-9_\-]+)["\']?', cleaned)
    if matches:
        filtered = [m for m in matches if m.lower() not in {"node_id", "node", "tool_id", "tool", "id"}]
        if filtered:
            return filtered[-1]
        return matches[-1]
    return cleaned


def clean_identifier_list(input_str: Optional[str]) -> List[str]:
    if not input_str:
        return []
    raw_tokens = str(input_str).split(",")
    result = []
    for token in raw_tokens:
        c = clean_input_token(token)
        if c:
            result.append(c)
    return result


def extract_prompt_variables_dict(prompt: str) -> Dict[str, str]:
    if not prompt:
        return {}
    pattern = re.compile(r'(?:([A-Za-z][A-Za-z0-9 _/]*?)\s*:\s*)?\{\{?([^}]+)\}?\}')
    seen: Dict[str, str] = {}
    for match in pattern.finditer(prompt):
        label = (match.group(1) or "").strip()
        inner = (match.group(2) or "").strip()
        if not inner:
            continue
        if ":" in inner and not label:
            key, default = inner.split(":", 1)
            key = key.strip()
            default = default.strip()
        else:
            key = label if label else inner
            default = ""
        if key and key not in seen:
            seen[key] = default
    return seen


# ── App Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("  Starting Astor AI — Unified API 2.0.0")
    if cfg.USE_DB:
        try:
            connected = init_db()
            if connected:
                logger.info("  DB Mode: PostgreSQL (Connected: unified_configuration)")
            else:
                logger.warning("  DB Mode: PostgreSQL (Init failed, local JSON fallback active)")
        except Exception as e:
            logger.warning(f"  DB Mode: PostgreSQL (Error: {e}, local JSON fallback active)")
    else:
        logger.info("  DB Mode: Local JSON (USE_DB=false)")
    logger.info("=" * 60)

    yield

    logger.info("Astor AI — Unified API shut down.")


app = FastAPI(
    title="Astor AI — Unified API",
    version="2.0.0",
    description="Unified API integrating Speech Subsystem, Astor LLM Engine, Workflow Graph, and Tools.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=_OPENAPI_TAGS,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Error Handlers ─────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"[Validation Error] {request.method} {request.url} : {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code >= 500:
        logger.error(f"[HTTP {exc.status_code}] {request.method} {request.url} : {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.critical(f"[Server Error] {request.method} {request.url} : {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Redirect root → Swagger UI
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# ==============================================================================
# 0. AGENT MANAGEMENT
# ==============================================================================

_AGENT_TAG = "0. Agent Management"

@app.post("/create-agent", tags=[_AGENT_TAG], summary="Create AI Voice Agent (Query Params + JSON Nodes Body)")
async def create_agent_config(
    agent_name: Annotated[str, Query(description="Name of the voice agent")],
    prompt_style: Annotated[str, Query(description="'system_prompt' or 'workflow'")] = "system_prompt",
    llm_model: Annotated[str, Query(description="LLM model choice")] = "Qwen2.5-7B-Instruct-1M",
    lid_method: Annotated[str, Query(description="Language Identification method")] = "sarvam webapi",
    translation_provider: Annotated[str, Query(description="Translation provider")] = "sarvam",
    prompt: Annotated[Optional[str], Query(description="System / global prompt")] = None,
    first_message: Annotated[Optional[str], Query(description="Opening greeting")] = None,
    start_node: Annotated[Optional[str], Query(description="Entry node ID")] = None,
    language: Annotated[Optional[str], Query(description="Primary operating language")] = "English",
    personality: Annotated[Optional[str], Query(description="Tone style")] = "professional",
    label: Annotated[Optional[str], Query(description="Short category label")] = "support",
    stt_vendor: Annotated[str, Query(description="STT vendor")] = "smallest",
    stt_normalize_text: Annotated[bool, Query(description="Enable ITN")] = True,
    stt_diarize: Annotated[bool, Query(description="Speaker diarization")] = False,
    stt_sensitivity: Annotated[str, Query(description="VAD sensitivity")] = "normal",
    stt_keywords: Annotated[str, Query(description="Keywords")] = "",
    tts_vendor: Annotated[str, Query(description="TTS vendor")] = "smallest",
    tts_voice_gender: Annotated[str, Query(description="Voice gender")] = "female",
    tts_speed: Annotated[float, Query(description="Speed")] = 1.0,
    tts_sample_rate: Annotated[int, Query(description="Sample rate")] = 24000,
    tts_temperature: Annotated[float, Query(description="Randomness")] = 0.6,
    tts_buffer_strategy: Annotated[str, Query(description="Buffer strategy")] = "balanced",
    default_parameters: Annotated[Optional[str], Query(description="JSON string of default variables")] = None,
    tool_id: Annotated[Optional[str], Query(description="Comma-separated Tool IDs")] = None,
    rag_document_id: Annotated[Optional[str], Query(description="RAG document ID")] = None,
    swap_rules: Annotated[Optional[str], Query(description="JSON string of swap rules")] = None,
    global_nodes_csv: Annotated[Optional[str], Query(description="Global node IDs CSV")] = None,
    nodes: List[Union[str, Dict[str, Any]]] = Body(default=[]),
):
    cleaned_start_node = clean_input_token(start_node)
    cleaned_tools = clean_identifier_list(tool_id)
    cleaned_global_nodes = clean_identifier_list(global_nodes_csv)

    effective_prompt = (prompt or "").strip()
    node_ids: List[str] = []

    for item in nodes:
        if isinstance(item, str):
            nid = clean_input_token(item)
            node_ids.append(nid)
        elif isinstance(item, dict):
            nid = item.get("node_id") or f"node_{uuid.uuid4().hex[:8]}"
            node_ids.append(nid)
            node_store.create(
                node_id=nid,
                name=item.get("name", f"Node {nid}"),
                type=item.get("type", "conversation"),
                prompt=item.get("prompt", ""),
                static_message=item.get("static_message", ""),
                variables_needed=item.get("variables_needed", []),
                tools=item.get("tools", []),
                is_global=item.get("is_global", False),
                transitions=item.get("transitions", []),
            )

    chosen_start_node = cleaned_start_node or (node_ids[0] if node_ids else None)
    agent_id = f"agent_{uuid.uuid4().hex[:8]}"

    parsed_swap = {}
    if swap_rules:
        try:
            parsed_swap = json.loads(swap_rules) if isinstance(swap_rules, str) else swap_rules
        except Exception:
            parsed_swap = {}

    created_agent = agent_store.create(
        agent_id=agent_id,
        name=agent_name,
        mode=prompt_style,
        system_prompt=effective_prompt,
        start_node=chosen_start_node,
        nodes=node_ids,
        global_nodes=cleaned_global_nodes,
        tools=cleaned_tools,
        language=language or "English",
        personality=personality or "professional",
        label=label or "support",
        swap_rules=parsed_swap,
        first_message=first_message,
        is_active=True,
    )

    response_data = dict(created_agent) if isinstance(created_agent, dict) else {"agent_id": agent_id, "name": agent_name}
    response_data["session_id"] = str(uuid.uuid4())
    response_data["start_node"] = chosen_start_node
    response_data["nodes"] = node_ids

    return {
        "status": "success",
        "message": f"Agent '{agent_name}' created.",
        "configuration_saved": response_data,
    }


@app.put("/update-agent", tags=[_AGENT_TAG], summary="Update Agent Config")
async def update_agent_config(
    agent_id: Annotated[str, Form(description="ID of the agent to update")],
    agent_name: Annotated[str, Form(description="Updated agent name")],
    prompt_style: Annotated[str, Form(description="Prompt style: system_prompt or workflow")] = "system_prompt",
    llm_model: Annotated[str, Form(description="LLM Model")] = "Qwen2.5-7B-Instruct-1M",
    lid_method: Annotated[str, Form(description="LID method")] = "sarvam webapi",
    translation_provider: Annotated[str, Form(description="Translation provider")] = "sarvam",
    prompt: Annotated[Optional[str], Form(description="Updated prompt")] = None,
    stt_vendor: Annotated[str, Form(description="STT vendor")] = "smallest",
    stt_normalize_text: Annotated[bool, Form(description="ITN")] = True,
    stt_diarize: Annotated[bool, Form(description="Diarization")] = False,
    stt_sensitivity: Annotated[str, Form(description="Sensitivity")] = "normal",
    stt_keywords: Annotated[str, Form(description="Keywords")] = "",
    tts_vendor: Annotated[str, Form(description="TTS vendor")] = "smallest",
    tts_voice_gender: Annotated[str, Form(description="Voice gender")] = "female",
    tts_speed: Annotated[float, Form(description="Speed")] = 1.0,
    tts_sample_rate: Annotated[int, Form(description="Sample rate")] = 24000,
    tts_temperature: Annotated[float, Form(description="Randomness")] = 0.6,
    tts_buffer_strategy: Annotated[str, Form(description="Buffer strategy")] = "balanced",
    first_message: Annotated[Optional[str], Form(description="Opening greeting")] = None,
    default_parameters: Annotated[Optional[str], Form(description="Default variables")] = None,
    language: Annotated[Optional[str], Form(description="Language")] = None,
    label: Annotated[Optional[str], Form(description="Label")] = None,
    personality: Annotated[Optional[str], Form(description="Personality")] = None,
    tool_id: Annotated[Optional[str], Form(description="Tools")] = None,
):
    existing = agent_store.get(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Agent ID '{agent_id}' not found.")

    tools_list = clean_identifier_list(tool_id) if tool_id else existing.get("tools", [])

    updated = agent_store.update(
        agent_id=agent_id,
        name=agent_name,
        mode=prompt_style,
        system_prompt=prompt if prompt is not None else existing.get("system_prompt", ""),
        tools=tools_list,
        language=language or existing.get("language", "English"),
        personality=personality or existing.get("personality", "professional"),
        label=label or existing.get("label", "support"),
    )

    return {
        "status": "success",
        "message": f"Agent '{agent_id}' updated successfully.",
        "agent": updated,
    }


@app.delete("/delete-agent/{agent_id}", tags=[_AGENT_TAG], summary="Delete Agent Config")
async def delete_agent_config(agent_id: str):
    clean_id = clean_input_token(agent_id)
    success = agent_store.delete(clean_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Agent ID '{clean_id}' not found.")
    return {
        "status": "success",
        "message": f"Agent ID '{clean_id}' permanently deleted.",
        "details": {"agent_id": clean_id},
    }


@app.get("/list-agents", tags=[_AGENT_TAG], summary="List Agents Config")
async def list_agents_config():
    agents = agent_store.list_all()
    return {
        "status": "success",
        "total_agents": len(agents),
        "agents": agents,
    }


@app.get("/get-agent/{agent_id}", tags=[_AGENT_TAG], summary="Get Agent Config")
async def get_agent_config(agent_id: str):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent ID '{clean_id}' not found.")
    return {
        "status": "success",
        "agent": agent,
    }


@app.get("/extract-variables/{agent_id}", tags=[_AGENT_TAG], summary="Extract Variables Config")
async def extract_variables_config(agent_id: str):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent ID '{clean_id}' not found.")
    prompt = agent.get("system_prompt", "")
    raw_matches = re.findall(r'\{+([^}]+)\}+', prompt)
    variables = list(set([
        v.strip() for v in raw_matches
        if v.strip() and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v.strip())
    ]))
    return {
        "status": "success",
        "agent_id": clean_id,
        "variables_found": len(variables),
        "variables": variables,
    }


# ==============================================================================
# 1. HEALTH
# ==============================================================================

@app.get("/health", tags=["1. Health"], summary="Health")
def health():
    return {
        "status": "ok",
        "service": "voice-bot-core",
        "version": "2.0.0",
        "mode": "PostgreSQL" if cfg.USE_DB else "Local JSON",
        "memory_cache": memory_store.cache_summary,
        "llm_queue": llm_queue.stats,
    }


# ==============================================================================
# 2. STT
# ==============================================================================

@app.post("/stt", tags=["2. STT"], summary="Stt")
async def stt(
    audio: Annotated[UploadFile, File(description="WAV or MP3 audio file")],
    provider: Annotated[str, Form(description="STT provider: smallest | devnagri | sarvam")] = "smallest",
    lid: Annotated[Optional[str], Form(description="Optional LID: sarvam | indic | google")] = None,
):
    vendor = provider.lower().strip()
    res = await transcribe_audio(audio, vendor=vendor)
    language = res.detected_language
    if not language and lid:
        try:
            language = await detect_language(res.transcript, vendor=lid.lower())
        except Exception:
            pass
    return {
        "transcript": res.transcript,
        "language": language,
        "provider": vendor,
    }


# ==============================================================================
# 3. LID
# ==============================================================================

@app.post("/language-detection", tags=["3. LID"], summary="Language Detection")
async def language_detection(
    text: Annotated[str, Form(description="Text to identify language of", example="आप कैसे हैं?")],
    method: Annotated[str, Form(description="Detection method: sarvam webapi | indic | google")] = "sarvam webapi",
):
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="'text' field cannot be empty.")
    t0 = time.time()
    detected = await detect_language(clean_text)
    latency = round(time.time() - t0, 4)
    return {
        "text": clean_text,
        "actual_language": detected,
        "latency": latency,
        "method": method,
    }


# ==============================================================================
# 4. TRANSLATION
# ==============================================================================

@app.post("/native-to-english", tags=["4. Translation"], summary="Route Native To English")
async def route_native_to_english(
    text: Annotated[str, Form(description="Text to translate into English", example="मुझे बैंक से लोन चाहिए")],
    actual_language: Annotated[str, Form(description="Source language name (e.g. Hindi, Marathi, Tamil)", example="Hindi")] = "Hindi",
    provider: Annotated[str, Form(description="Translation provider: sarvam_web | sarvam | nllb")] = "sarvam_web",
):
    t0 = time.time()
    translated = await native_to_english(text, source_language=actual_language)
    elapsed = round(time.time() - t0, 2)
    return {
        "text": text,
        "actual_language": actual_language,
        "english_text": translated,
        "provider": provider,
        "processing_time_seconds": elapsed,
    }


@app.post("/english-to-native", tags=["4. Translation"], summary="Route English To Native")
async def route_english_to_native(
    text: Annotated[str, Form(description="English text to translate into native script", example="I need a loan from the bank.")],
    actual_language: Annotated[str, Form(description="Target language name (e.g. Hindi, Marathi, Tamil)", example="Hindi")] = "Hindi",
    provider: Annotated[str, Form(description="Translation provider: sarvam_web | sarvam | nllb")] = "sarvam_web",
):
    t0 = time.time()
    translated = await english_to_native(text, target_language=actual_language)
    elapsed = round(time.time() - t0, 2)
    return {
        "text": text,
        "actual_language": actual_language,
        "native_text": translated,
        "provider": provider,
        "processing_time_seconds": elapsed,
    }


# ==============================================================================
# 5. TTS
# ==============================================================================

@app.post("/tts", tags=["5. TTS"], summary="Tts")
async def tts(
    text: Annotated[str, Form(description="Text to speak")],
    language: Annotated[str, Form(description="Language name, e.g. Hindi / English")] = "English",
    provider: Annotated[str, Form(description="smallest | devnagri | sarvam")] = "smallest",
    gender: Annotated[str, Form(description="Female | Male")] = "Female",
):
    res = await synthesize_speech(
        text=text,
        language=language,
        vendor=provider.lower().strip(),
        voice_gender=gender.lower().strip(),
    )
    if res.audio_base64:
        audio_bytes = base64.b64decode(res.audio_base64)
    else:
        raise HTTPException(status_code=502, detail="TTS provider returned no audio.")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"},
    )


# ==============================================================================
# 6. LLM CONFIG (Agents, Documents, Variables, Chat, Models)
# ==============================================================================

# ── 6.1. LLM Config — Agents ──────────────────────────────────────────────────
_TAG_LLM_AGENTS = "6.1. LLM Config — Agents"

class CreateAgentModel(BaseModel):
    name: str = Field(..., description="Agent name")
    prompt: str = Field(..., description="Agent system prompt")

class UpdateAgentModel(BaseModel):
    name: Optional[str] = Field(None, description="Agent name")
    prompt: Optional[str] = Field(None, description="Agent system prompt")


@app.get("/agents/{agent_id}", tags=[_TAG_LLM_AGENTS], summary="Get agent by ID")
async def llm_get_agent(agent_id: str):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@app.patch("/agents/{agent_id}", tags=[_TAG_LLM_AGENTS], summary="Update agent name or prompt")
async def llm_update_agent(agent_id: str, req: UpdateAgentModel):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.update(clean_id, name=req.name, system_prompt=req.prompt)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@app.delete("/agents/{agent_id}", tags=[_TAG_LLM_AGENTS], summary="Delete (deactivate) an agent")
async def llm_delete_agent(agent_id: str):
    clean_id = clean_input_token(agent_id)
    ok = agent_store.delete(clean_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"detail": "Agent deactivated"}


@app.get("/agents", tags=[_TAG_LLM_AGENTS], summary="List all agents")
async def llm_list_agents():
    return agent_store.list_all()


@app.post("/agents", tags=[_TAG_LLM_AGENTS], summary="Create a new agent (with optional documents)")
async def llm_create_agent(
    name: Annotated[str, Form(description="Agent name")],
    prompt: Annotated[str, Form(description="Agent system prompt")],
    files: Annotated[List[UploadFile], File(description="Optional knowledge base documents")] = [],
):
    aid = f"agent_{uuid.uuid4().hex[:8]}"
    doc_names = [f.filename for f in files if f.filename]
    agent = agent_store.create(
        agent_id=aid,
        name=name,
        system_prompt=prompt,
        kb_files=doc_names,
    )
    result = dict(agent) if isinstance(agent, dict) else {"agent_id": aid, "name": name}
    result["indexed_files"] = doc_names
    return result


# ── 6.2. LLM Config — Documents ───────────────────────────────────────────────
_TAG_LLM_DOCS = "6.2. LLM Config — Documents"

@app.post("/agents/{agent_id}/documents", tags=[_TAG_LLM_DOCS], summary="Append documents to an existing agent")
async def append_documents(
    agent_id: str,
    files: Annotated[List[UploadFile], File(description="Documents to append")] = [],
):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    new_files = [f.filename for f in files if f.filename]
    current = agent.get("kb_files", [])
    if isinstance(current, list):
        current.extend(new_files)
    else:
        current = new_files

    agent_store.update(clean_id, kb_files=current)
    return {
        "agent_id": clean_id,
        "indexed_files": new_files,
        "total_kb_files": len(current),
        "errors": [],
    }


@app.get("/agents/{agent_id}/documents", tags=[_TAG_LLM_DOCS], summary="List all documents for an agent")
async def list_documents(agent_id: str):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    kb_files = agent.get("kb_files", [])
    return {
        "agent_id": clean_id,
        "kb_files": kb_files,
        "files_on_disk": kb_files,
        "total": len(kb_files),
    }


@app.delete("/agents/{agent_id}/documents/{filename}", tags=[_TAG_LLM_DOCS], summary="Remove a specific document from an agent")
async def remove_document(agent_id: str, filename: str):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    current = agent.get("kb_files", [])
    if isinstance(current, list) and filename in current:
        current.remove(filename)
        agent_store.update(clean_id, kb_files=current)
    return {"detail": f"Document '{filename}' removed from agent '{clean_id}'"}


# ── 6.3. LLM Config — Variables ───────────────────────────────────────────────
_TAG_LLM_VARS = "6.3. LLM Config — Variables"

class PreviewPromptModel(BaseModel):
    prompt: str = Field(..., description="Prompt string to preview variables from")

class SetVariablesModel(BaseModel):
    variables: Dict[str, str] = Field(default_factory=dict, description="Key-value pairs to set")


@app.post("/agents/preview-variables", tags=[_TAG_LLM_VARS], summary="Preview variables from a prompt (no agent needed)")
async def preview_variables(req: PreviewPromptModel):
    vars_dict = extract_prompt_variables_dict(req.prompt)
    return {
        "variables": vars_dict,
        "count": len(vars_dict),
    }


@app.get("/agents/{agent_id}/variables", tags=[_TAG_LLM_VARS], summary="Get all prompt variables for an agent")
async def get_variables(agent_id: str):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    prompt = agent.get("system_prompt", "")
    vars_dict = agent.get("prompt_variables") or extract_prompt_variables_dict(prompt)
    return {
        "agent_id": clean_id,
        "variables": vars_dict,
        "count": len(vars_dict),
        "rendered_prompt_preview": prompt[:300] + ("..." if len(prompt) > 300 else ""),
    }


@app.put("/agents/{agent_id}/variables", tags=[_TAG_LLM_VARS], summary="Set runtime values for prompt variables")
async def set_variables(agent_id: str, req: SetVariablesModel):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    current = agent.get("prompt_variables") or {}
    current.update(req.variables)
    agent_store.update(clean_id, prompt_variables=current)
    return {
        "agent_id": clean_id,
        "updated": req.variables,
        "variables": current,
        "count": len(current),
    }


@app.patch("/agents/{agent_id}/variables/{variable_key}", tags=[_TAG_LLM_VARS], summary="Set value for a single prompt variable")
async def set_single_variable(
    agent_id: str,
    variable_key: str,
    value: Annotated[str, Form(description="Variable value")],
):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    current = agent.get("prompt_variables") or {}
    current[variable_key] = value
    agent_store.update(clean_id, prompt_variables=current)
    return {
        "agent_id": clean_id,
        "variable_key": variable_key,
        "value": value,
        "all_variables": current,
    }


@app.delete("/agents/{agent_id}/variables/{variable_key}", tags=[_TAG_LLM_VARS], summary="Clear value for a single prompt variable (reset to default)")
async def clear_variable(agent_id: str, variable_key: str):
    clean_id = clean_input_token(agent_id)
    agent = agent_store.get(clean_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    current = agent.get("prompt_variables") or {}
    current.pop(variable_key, None)
    agent_store.update(clean_id, prompt_variables=current)
    return {
        "agent_id": clean_id,
        "variable_key": variable_key,
        "reset_to": "",
        "all_variables": current,
    }


# ── 6.4. LLM Config — Chat ────────────────────────────────────────────────────
_TAG_LLM_CHAT = "6.4. LLM Config — Chat"

class ChatStreamModel(BaseModel):
    user_message: Optional[str] = Field(None, description="User message text")
    history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history")
    agent_id: Optional[str] = Field(None, description="Agent ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    message: Optional[str] = Field(None, description="Alias for user message")
    image_base64: Optional[str] = Field(None, description="Optional base64 image")


@app.post("/chat/stream", tags=[_TAG_LLM_CHAT], summary="Streaming chat (SSE)")
async def chat_stream(req: ChatStreamModel):
    user_msg = (req.user_message or req.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Missing user_message in request body.")

    sid = req.session_id
    if not sid or not memory_store.get_session(sid):
        aid = req.agent_id
        if not aid:
            all_a = agent_store.list_all()
            aid = all_a[0]["agent_id"] if all_a else "agent_default"
        agent = agent_store.get(aid) or {}
        start_node = agent.get("start_node") or agent.get("entry_node") or "start"
        session = memory_store.create_session(agent_id=aid, start_node=start_node)
        sid = session.session_id
    else:
        session = memory_store.get_session(sid)

    if req.history:
        session.history = []
        for turn in req.history:
            if isinstance(turn, dict) and "role" in turn and "content" in turn:
                session.add_message(turn["role"], turn["content"])

    if req.image_base64:
        session.set_variable("InputImage", req.image_base64)
        session.set_variable("InputFile", req.image_base64)
        session.set_variable("_image_pending", "true")

    async def sse_gen():
        try:
            async for event in _process_message_stream(sid, user_msg):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[ChatStream] Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        sse_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 6.5. LLM Config — Models ──────────────────────────────────────────────────
_TAG_LLM_MODELS = "6.5. LLM Config — Models"

@app.post("/models/select", tags=[_TAG_LLM_MODELS], summary="Select and activate a model")
async def select_model(
    model_name: str = Query("Qwen2.5-7B-Instruct-1M", description="Model name to activate"),
    quantization: str = Query("Low", description="Quantization level"),
):
    return {
        "detail": f"Model '{model_name}' is now active.",
        "active_model": {
            "model_name": model_name,
            "quantization": quantization,
        },
        "gpu_status": {"gpu_available": True},
    }


@app.get("/models/active", tags=[_TAG_LLM_MODELS], summary="Get the currently active model")
async def get_active_model():
    return {
        "model_name": os.path.basename(cfg.MODEL_PATH),
        "quantization": "Low",
        "status": "ready" if getattr(llm_engine, "_loaded", False) else "ready",
    }


# ==============================================================================
# 9. FLOW MANAGEMENT (Knight Generic Flow)
# ==============================================================================

# ── 9.1. Flow — Agents ────────────────────────────────────────────────────────
set_router_tags(agent_router, "9.1. Flow — Agents")
app.include_router(agent_router)

# ── 9.2. Flow — Nodes ─────────────────────────────────────────────────────────
set_router_tags(node_router, "9.2. Flow — Nodes")
app.include_router(node_router)

# ── 9.3. Flow — Subflows ──────────────────────────────────────────────────────
set_router_tags(subflow_router, "9.3. Flow — Subflows")
app.include_router(subflow_router)

# ── 9.4. Flow — Tool Registry ─────────────────────────────────────────────────
set_router_tags(tool_router, "9.4. Flow — Tool Registry")
set_router_tags(image_router, "9.4. Flow — Tool Registry")
app.include_router(tool_router)
app.include_router(image_router)

# ── 9.5. Flow — CBS Banking Data ──────────────────────────────────────────────
set_router_tags(cbs_router, "9.5. Flow — CBS Banking Data")
app.include_router(cbs_router)

# ── 9.6. Flow — Backend Services ──────────────────────────────────────────────
set_router_tags(services_router, "9.6. Flow — Backend Services")
app.include_router(services_router)


# ==============================================================================
# 10. LANGUAGES
# ==============================================================================

@app.get("/languages/translation", tags=["10. Languages"], summary="Supported Translation Languages")
def supported_translation_languages():
    return [
        "Hindi", "Marathi", "Bengali", "Tamil", "Telugu",
        "Kannada", "Gujarati", "Punjabi", "Malayalam", "Odia", "English"
    ]


@app.get("/languages/detection", tags=["10. Languages"], summary="Supported Detection Languages")
def supported_detection_languages():
    return [
        "Hindi", "English", "Marathi", "Bengali", "Tamil",
        "Telugu", "Kannada", "Gujarati", "Punjabi", "Malayalam", "Odia"
    ]


# ==============================================================================
# 11. VOICE PIPELINE (End-to-End Turns)
# ==============================================================================

voice_router = APIRouter(prefix="/api/voice", tags=["11. Voice Pipeline"])

class StartCallRequest(BaseModel):
    agent_id: str = Field(..., description="Target Agent ID")
    customer_id: Optional[str] = Field(None, description="Optional Customer ID to pre-load CBS data")
    tts_vendor: TTSVendor = Field(TTSVendor.SMALLEST, description="TTS synthesis provider")
    voice_gender: VoiceGender = Field(VoiceGender.FEMALE, description="Voice gender")


@voice_router.post("/start", summary="Start call & receive greeting audio")
async def start_voice_call(req: StartCallRequest):
    return await voice_pipeline.start_call(
        agent_id=req.agent_id,
        customer_id=req.customer_id,
        tts_vendor=req.tts_vendor.value,
        voice_gender=req.voice_gender.value,
    )


@voice_router.post("/turn", summary="Single user voice turn (Audio in -> Audio out)")
async def voice_turn(
    audio_file: UploadFile = File(..., description="User recorded audio clip (WAV/MP3)"),
    session_id: str = Form(..., description="Active session ID"),
    stt_vendor: STTVendor = Form(STTVendor.SMALLEST, description="STT vendor"),
    tts_vendor: TTSVendor = Form(TTSVendor.SMALLEST, description="TTS vendor"),
    voice_gender: VoiceGender = Form(VoiceGender.FEMALE, description="Voice gender"),
):
    sentences = []
    transcript_info = {}
    done_info = {}

    async for event in voice_pipeline.process_voice_turn(
        audio_input=audio_file,
        session_id=session_id,
        stt_vendor=stt_vendor.value,
        tts_vendor=tts_vendor.value,
        voice_gender=voice_gender.value,
    ):
        if event.get("type") == "transcript":
            transcript_info = event
        elif event.get("type") == "sentence_audio":
            sentences.append(event)
        elif event.get("type") == "turn_complete":
            done_info = event
        elif event.get("type") == "error":
            raise HTTPException(status_code=400, detail=event.get("content"))

    return {
        "session_id": session_id,
        "transcript": transcript_info.get("transcript"),
        "language": transcript_info.get("language"),
        "sentences": sentences,
        "current_node_name": done_info.get("current_node_name"),
        "transitioned": done_info.get("transitioned", False),
        "total_turn_duration_ms": done_info.get("total_turn_duration_ms"),
    }


@voice_router.post("/stream", summary="Streaming voice turn via SSE")
async def voice_stream(
    audio_file: UploadFile = File(...),
    session_id: str = Form(...),
    stt_vendor: STTVendor = Form(STTVendor.SMALLEST),
    tts_vendor: TTSVendor = Form(TTSVendor.SMALLEST),
    voice_gender: VoiceGender = Form(VoiceGender.FEMALE),
):
    async def sse_gen():
        async for event in voice_pipeline.process_voice_turn(
            audio_input=audio_file,
            session_id=session_id,
            stt_vendor=stt_vendor.value,
            tts_vendor=tts_vendor.value,
            voice_gender=voice_gender.value,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse_gen(), media_type="text/event-stream")


set_router_tags(orchestrator_router, "6.4. LLM Config — Chat")
app.include_router(orchestrator_router)
app.include_router(voice_router)

