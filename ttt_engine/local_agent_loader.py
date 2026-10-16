from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from config import cfg
from logger import get_logger

logger = get_logger(__name__)

_loaded: bool = False
_agents: Dict[str, Dict] = {}
_raw_config: Optional[Dict] = None


def _agents_file_path() -> str:
    return os.path.join(cfg.DATA_DIR, "agents.json")


def _save_agents_to_file() -> None:
    path = _agents_file_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(list(_agents.values()), fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error(f"[LocalAgentLoader] Failed to save {path}: {exc}")


def _load_once() -> None:
    global _loaded, _agents, _raw_config
    if _loaded:
        return

    # 1. Load data/agents.json if it exists
    agents_file = _agents_file_path()
    if os.path.isfile(agents_file):
        try:
            with open(agents_file, "r", encoding="utf-8") as fh:
                items = json.load(fh)
                if isinstance(items, list):
                    for item in items:
                        aid = item.get("agent_id") or item.get("id")
                        if aid:
                            item["agent_id"] = aid
                            _agents[aid] = item
            logger.info(f"[LocalAgentLoader] Loaded {len(_agents)} agents from {agents_file}")
        except Exception as exc:
            logger.warning(f"[LocalAgentLoader] Failed to parse {agents_file}: {exc}")

    # 2. Also load LOCAL_AGENT_CONFIG_PATH
    path = cfg.LOCAL_AGENT_CONFIG_PATH
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                _raw_config = json.load(fh)
            agent_id = _raw_config.get("agent_id") or "local_agent"
            flow = _raw_config.get("conversationFlow", {})
            nodes_list = flow.get("nodes", [])
            global_node_ids = flow.get("global_node_ids", [])
            all_node_ids = [n["id"] for n in nodes_list]
            tools_list = _raw_config.get("tools", [])
            tool_ids = [t["tool_id"] for t in tools_list if "tool_id" in t]
            agent_dict = {
                "agent_id": agent_id,
                "name": _raw_config.get("agent_name", "Local Agent"),
                "mode": _raw_config.get("mode", "flow"),
                "system_prompt": (_raw_config.get("system_prompt") or flow.get("global_prompt", "")),
                "start_node": flow.get("start_node_id", all_node_ids[0] if all_node_ids else ""),
                "nodes": all_node_ids,
                "global_nodes": global_node_ids,
                "tools": tool_ids,
                "language": _raw_config.get("language", "english"),
                "personality": _raw_config.get("personality"),
                "swap_rules": _raw_config.get("swap_rules", {}),
                "agent_config_json": _raw_config,
                "created_at": "",
                "updated_at": "",
            }
            _agents[agent_id] = agent_dict
            try:
                from tool_registry import tool_store as _tool_store
                for tool in tools_list:
                    tid = tool.get("tool_id")
                    if tid:
                        _tool_store.register_local_tool(tool)
            except Exception as exc:
                logger.warning(f"[LocalAgentLoader] Could not register tools: {exc}")
        except Exception as exc:
            logger.error(f"[LocalAgentLoader] Failed to parse {path}: {exc}")

    _loaded = True


class LocalAgentStore:
    def get(self, agent_id: str) -> Optional[Dict]:
        _load_once()
        return _agents.get(agent_id)

    def list_all(self) -> List[Dict]:
        _load_once()
        return list(_agents.values())

    def create(self, **kwargs) -> Dict:
        _load_once()
        agent_id = kwargs.get("agent_id") or f"agent_{len(_agents) + 1:02d}"
        agent_dict = {
            "agent_id": agent_id,
            "name": kwargs.get("name", "New Agent"),
            "mode": kwargs.get("mode", "flow"),
            "system_prompt": kwargs.get("system_prompt", kwargs.get("prompt", "")),
            "start_node": kwargs.get("start_node"),
            "nodes": kwargs.get("nodes", []),
            "global_nodes": kwargs.get("global_nodes", []),
            "tools": kwargs.get("tools", []),
            "language": kwargs.get("language", "English"),
            "personality": kwargs.get("personality", "professional"),
            "label": kwargs.get("label", "support"),
            "swap_rules": kwargs.get("swap_rules", {}),
            "is_active": kwargs.get("is_active", True),
            "agent_config_json": kwargs.get("agent_config_json", {}),
        }
        _agents[agent_id] = agent_dict
        _save_agents_to_file()
        logger.info(f"[LocalAgentLoader] Created agent '{agent_dict['name']}' (id={agent_id})")
        return agent_dict

    def update(self, agent_id: str, **kwargs) -> Optional[Dict]:
        _load_once()
        agent = _agents.get(agent_id)
        if not agent:
            return None
        for k, v in kwargs.items():
            if v is not None:
                agent[k] = v
        _save_agents_to_file()
        logger.info(f"[LocalAgentLoader] Updated agent '{agent_id}'")
        return agent

    def delete(self, agent_id: str) -> bool:
        _load_once()
        if agent_id in _agents:
            del _agents[agent_id]
            _save_agents_to_file()
            logger.info(f"[LocalAgentLoader] Deleted agent '{agent_id}'")
            return True
        return False


local_agent_store = LocalAgentStore()
