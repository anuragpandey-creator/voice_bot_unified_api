# Voice Bot Core — Dynamic Conversational Voice Platform

A unified, turnkey voice bot engine combining:
* **Speech Subsystem**: Speech-to-Text (STT), Language Identification (LID), Bidirectional Translation, and Text-to-Speech (TTS).
* **TTT Engine**: The Astor LLM Orchestration Layer with KV-cached prompt reuse, sentence-level streaming, concurrency queuing (`llm_queue`), and state machine transitions.
* **Dynamic Tool Registry**: Onboard and execute any external REST API (including external RAG services) with `{Variable}` request templating and JSON response mapping.
* **Dual Execution Mode**: Supports both PostgreSQL database storage (`USE_DB=true`) and standalone offline mode (`USE_DB=false`).

---

## Architecture

```
User Audio (Mic/Upload)
         │
         ▼
[speech/stt.py]            → Smallest AI / Sarvam / Devnagri STT
         │
         ▼
[speech/translation.py]    → Native to English Translation
         │
         ▼
[ttt_engine/orchestrator]  → Edge Transitions & Dynamic Tool / CBS / RAG Calls
         │
         ▼
[ttt_engine/llm_engine]    → KV-Cached Qwen Generation (Sentence Streaming)
         │
         ▼
[speech/translation.py]    → English to Native Translation
         │
         ▼
[speech/tts.py]            → Smallest AI / Sarvam / Devnagri Audio Synthesis
         │
         ▼
Bot Audio Response (Speaker)
```

---

## How to Run

### 1. Start the Backend API (Terminal 1)
Run `main:app` directly using the `uvicorn` command with your desired port:
```bash
# Example with port 8000 (or any port of your choice):
uvicorn main:app --port 1224

# With hot-reload during development:
uvicorn main:app  --port 1224 --reload
```
* **Interactive API Docs (Swagger)**: http://localhost:<YOUR_PORT>/docs
* **Health Check**: http://localhost:<YOUR_PORT>/health

### 2. Start the Gradio Demo Console (Terminal 2)
```bash
python gradio_app.py
```
* **Gradio Demo UI**: http://localhost:7870

---

## Gradio Console Tabs

1. **🔧 Tool Registry**:
   * Dynamically register external HTTP APIs.
   * Configure `{Variable}` placeholders in URL, headers, and request body.
   * Configure response variable mapping via JSON-path or template matching.
   * Pre-configured with the `query_knowledge_base` external RAG tool.
2. **🤖 Agent Builder**:
   * Configure Retell-style conversational agents.
   * Build conversation phases (nodes), instructions, static messages, and edge transitions inline.
3. **💬 Text Chat**:
   * Chat directly with any flow agent via text and SSE streaming.
4. **🎙️ Live Voice Bot Demo**:
   * Select an Agent and Customer.
   * Click **"Start Voice Call"** to hear the bot's synthesized greeting.
   * Record into the microphone or upload an audio clip to converse in real-time!

---

## Folder Structure

```
voice_bot_core/
├── .env                       # Unified environment variables
├── config.py                  # Central configuration loader
├── logger.py                  # Loguru logging setup
├── requirements.txt           # Unified dependency manifest
├── main.py                    # FastAPI backend server
├── gradio_app.py              # Single demo UI console
│
├── speech/                    # Speech & Language Processing
│   ├── stt.py                 # Smallest, Sarvam, Devnagri STT
│   ├── tts.py                 # Smallest, Sarvam, Devnagri TTS
│   ├── lid.py                 # Language identification
│   ├── translation.py         # Native ↔ English translation
│   └── models.py              # Pydantic schemas and enums
│
├── ttt_engine/                # Astor LLM Layer (The Orchestrator Brain)
│   ├── orchestrator.py        # State machine, node transitions, dynamic context
│   ├── llm_engine.py          # KV-cached Qwen, sentence streaming generator, GPU queue
│   ├── memory_store.py        # Session memory & multi-turn history
│   ├── agent_manager.py       # Agent definitions & CRUD
│   ├── node_manager.py        # Conversation, function, & subagent nodes
│   ├── tool_registry.py       # Dynamic tool onboarding & execution
│   ├── cbs_api.py             # CBS customer data APIs
│   ├── database.py            # PostgreSQL connector (with USE_DB toggle)
│   ├── local_agent_loader.py  # Standalone zero-DB runner (reads agent_config.json)
│   └── error_logger.py        # Telemetry & error logging
│
├── pipeline/                  # Voice Pipeline Glue
│   └── voice_pipeline.py      # End-to-end Audio In → STT → TTT → TTS Audio Out
│
└── data/                      # Default offline data & configurations
    ├── nodes.json             # Conversation nodes
    ├── tools.json             # Registered tools (including RAG tool)
    ├── agents.json            # Agent configurations
    ├── agent_config.json      # Offline local agent config
    └── customer_master_data/  # Mock banking CSVs
```

---

## API Endpoints (Full Backwards Compatibility)

All original API endpoints have been preserved with identical route names, request parameters, and response contracts so any existing system or frontend can integrate with zero code changes:

### 1. Legacy Core Endpoints (Root Path)
| Endpoint | Method | Input | Response | Description |
| :--- | :--- | :--- | :--- | :--- |
| `/stt` | `POST` | `audio` (file), `provider`, `lid` | JSON `{transcript, language, provider}` | Speech-to-text audio transcription |
| `/tts` | `POST` | `text`, `language`, `provider`, `gender` | `audio/wav` binary stream | Text-to-speech audio synthesis download |
| `/language-detection` | `POST` | `text`, `method` | JSON `{text, actual_language, latency, method}` | Language identification |
| `/native-to-english` | `POST` | `text`, `actual_language`, `provider` | JSON `{text, english_text, actual_language, ...}` | Native language to English translation |
| `/english-to-native` | `POST` | `text`, `actual_language`, `provider` | JSON `{text, native_text, actual_language, ...}` | English to native language translation |
| `/languages/translation`| `GET` | — | JSON list of languages | Supported translation languages |
| `/languages/detection`  | `GET` | — | JSON list of languages | Supported detection languages |
| `/chat/stream` | `POST` | `user_message`, `agent_id`, `session_id` | `text/event-stream` (SSE) | LLM token streaming with dynamic TTT state machine |
| `/create-agent` | `POST` | Query params + JSON `nodes` body | JSON `{status, message, configuration_saved}` | Create AI voice agent (single prompt or workflow) |
| `/update-agent` | `PUT` | Form params | JSON `{status, message, agent}` | Update existing agent configuration |
| `/delete-agent/{id}` | `DELETE` | Path param `agent_id` | JSON `{status, message, details}` | Permanently delete agent |
| `/list-agents` | `GET` | — | JSON `{status, total_agents, agents}` | List all active agents |
| `/get-agent/{id}` | `GET` | Path param `agent_id` | JSON `{status, agent}` | Fetch agent details (also `/agents/{id}`) |
| `/extract-variables/{id}` | `GET` | Path param `agent_id` | JSON `{status, variables_found, variables}` | Extract `{variables}` from agent prompt |
| `/health` | `GET` | — | JSON `{status: "ok", mode, ...}` | Service health check |

### 2. Knight Flow & TTT Orchestration Endpoints
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/chat/start` | `POST` | Initialize conversation session and return opening greeting |
| `/api/chat/stream` | `POST` | Multi-turn streaming chat via SSE with transition tracking |
| `/api/agents` | `GET/POST` | Workflow agent management CRUD |
| `/api/nodes` | `GET/POST` | Workflow node graph CRUD & edge transitions |
| `/api/tools` | `GET/POST` | Dynamic tool registration & execution |
| `/api/cbs/*` | `GET` | Core Banking System customer & loan mock APIs |
| `/api/services/*` | `POST` | Banking verification, PTP agreement, and SMS services |
| `/api/subflows` | `GET/POST` | Reusable subflow modules |

### 3. Unified End-to-End Voice Pipeline
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/voice/start` | `POST` | Start outbound/inbound voice call and receive greeting audio |
| `/api/voice/turn` | `POST` | Single user voice turn (Audio In $\rightarrow$ STT $\rightarrow$ TTT $\rightarrow$ Audio Out) |
| `/api/voice/stream` | `POST` | SSE stream yielding sentence audio chunks in real-time |

---

## Connecting External RAG

The repository does not host local vector databases. Instead, the other team's RAG service is onboarded as a standard tool in `data/tools.json` or via the **Tool Registry** tab:

```json
{
  "tool_id": "query_knowledge_base",
  "name": "Query Knowledge Base (External RAG)",
  "type": "api_call",
  "method": "POST",
  "endpoint": "https://rag-service.internal/api/search",
  "request_body": {
    "query": "{UserMessage}",
    "agent_id": "{AgentId?}"
  },
  "response_body": {
    "answer": "{RAGAnswer}",
    "context": "{RAGContext}"
  }
}
```
When a customer asks a question requiring document knowledge, the orchestrator triggers this tool and injects the retrieved context directly into the prompt.
