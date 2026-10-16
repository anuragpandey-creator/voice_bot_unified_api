"""
node_manager.py — Node Data Layer (PostgreSQL)
======================================================
Nodes stored in PostgreSQL `nodes` table.

Nodes are no longer exposed as a standalone API — they're managed inline as
part of an agent's workflow (see agent_manager.py, which owns the
POST/GET/PUT/DELETE /api/nodes endpoints). This module just provides the
`node_store` data layer that agent_manager.py and orchestrator.py read from.
"""

from __future__ import annotations

import json
import uuid
from typing import Dict, List, Optional

import os
from config import cfg
from database import execute_query, execute_one, execute_write
from logger import get_logger

logger = get_logger(__name__)


# ── Node Store (PostgreSQL with Local JSON Fallback) ───────────────────────────

class NodeStore:
    """Node store supporting PostgreSQL or standalone local JSON file."""

    def __init__(self):
        self._local_nodes: Dict[str, Dict] = {}
        self._local_loaded: bool = False
        self._json_path = os.path.join(cfg.DATA_DIR, "nodes.json")

    def _ensure_local_loaded(self):
        if self._local_loaded:
            return
        if os.path.isfile(self._json_path):
            try:
                with open(self._json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            nid = item.get("node_id") or item.get("id")
                            if nid:
                                item["node_id"] = nid
                                self._local_nodes[nid] = item
                logger.info(f"[NodeStore] Loaded {len(self._local_nodes)} nodes from {self._json_path}")
            except Exception as e:
                logger.warning(f"[NodeStore] Failed to read {self._json_path}: {e}")
        self._local_loaded = True

    def _save_local(self):
        try:
            os.makedirs(os.path.dirname(self._json_path), exist_ok=True)
            with open(self._json_path, "w", encoding="utf-8") as f:
                json.dump(list(self._local_nodes.values()), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[NodeStore] Failed to save {self._json_path}: {e}")

    def create(self, **kwargs) -> Dict:
        node_id = kwargs.get("node_id") or f"node_{uuid.uuid4().hex[:8]}"

        if cfg.USE_DB:
            try:
                execute_write(
                    """INSERT INTO nodes (node_id, name, type, prompt, static_message, 
                       variables_needed, tools, is_global, max_turns, transitions)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        node_id,
                        kwargs["name"],
                        kwargs.get("type", "conversation"),
                        kwargs["prompt"],
                        kwargs.get("static_message"),
                        json.dumps(kwargs.get("variables_needed", [])),
                        json.dumps(kwargs.get("tools", [])),
                        kwargs.get("is_global", False),
                        kwargs.get("max_turns", 3),
                        json.dumps(kwargs.get("transitions", [])),
                    ),
                )
                logger.info(f"[NodeStore] Created in PostgreSQL: {node_id} ({kwargs['name']})")
                node = self.get(node_id)
                # Sync local fallback cache
                if node:
                    self._ensure_local_loaded()
                    self._local_nodes[node_id] = node
                    self._save_local()
                return node
            except Exception as e:
                logger.warning(f"[NodeStore] PostgreSQL create failed, falling back to local: {e}")

        # Local JSON fallback
        self._ensure_local_loaded()
        node_dict = {
            "node_id": node_id,
            "name": kwargs["name"],
            "type": kwargs.get("type", "conversation"),
            "prompt": kwargs["prompt"],
            "static_message": kwargs.get("static_message"),
            "variables_needed": kwargs.get("variables_needed", []),
            "tools": kwargs.get("tools", []),
            "is_global": kwargs.get("is_global", False),
            "max_turns": kwargs.get("max_turns", 3),
            "transitions": kwargs.get("transitions", []),
            "created_at": datetime.now().isoformat() if "datetime" in globals() else "",
        }
        self._local_nodes[node_id] = node_dict
        self._save_local()
        logger.info(f"[NodeStore] Created local node: {node_id} ({kwargs['name']})")
        return node_dict

    def get(self, node_id: str) -> Optional[Dict]:
        if cfg.USE_DB:
            try:
                row = execute_one("SELECT * FROM nodes WHERE node_id = %s", (node_id,))
                if row:
                    return self._format(row)
                return None
            except Exception as e:
                logger.warning(f"[NodeStore] DB error in get(), falling back to local: {e}")

        self._ensure_local_loaded()
        return self._local_nodes.get(node_id)

    def list_all(self) -> List[Dict]:
        if cfg.USE_DB:
            try:
                rows = execute_query("SELECT * FROM nodes ORDER BY created_at")
                if rows:
                    return [self._format(r) for r in rows]
            except Exception as e:
                logger.warning(f"[NodeStore] DB error in list_all(), falling back to local: {e}")

        self._ensure_local_loaded()
        return list(self._local_nodes.values())

    def update(self, node_id: str, **kwargs) -> Optional[Dict]:
        if cfg.USE_DB:
            try:
                existing = self.get(node_id)
                if not existing:
                    return None

                updates = []
                values = []
                for key in ("name", "type", "prompt", "static_message", "is_global", "max_turns"):
                    if kwargs.get(key) is not None:
                        updates.append(f"{key} = %s")
                        values.append(kwargs[key])

                for key in ("variables_needed", "tools", "transitions"):
                    if kwargs.get(key) is not None:
                        updates.append(f"{key} = %s")
                        values.append(json.dumps(kwargs[key]))

                if updates:
                    values.append(node_id)
                    execute_write(f"UPDATE nodes SET {', '.join(updates)} WHERE node_id = %s", tuple(values))
                    logger.info(f"[NodeStore] Updated in PostgreSQL: {node_id}")

                # Sync local cache
                node = self.get(node_id)
                if node:
                    self._ensure_local_loaded()
                    self._local_nodes[node_id] = node
                    self._save_local()
                return node
            except Exception as e:
                logger.warning(f"[NodeStore] DB error in update(), falling back to local: {e}")

        self._ensure_local_loaded()
        existing = self._local_nodes.get(node_id)
        if not existing:
            return None
        for k in ("name", "type", "prompt", "static_message", "variables_needed", "tools", "is_global", "max_turns", "transitions"):
            if k in kwargs and kwargs[k] is not None:
                existing[k] = kwargs[k]
        self._save_local()
        return existing

    def delete(self, node_id: str) -> bool:
        if cfg.USE_DB:
            try:
                existing = self.get(node_id)
                if not existing:
                    return False
                execute_write("DELETE FROM nodes WHERE node_id = %s", (node_id,))
                logger.info(f"[NodeStore] Deleted from PostgreSQL: {node_id}")
                self._ensure_local_loaded()
                if node_id in self._local_nodes:
                    del self._local_nodes[node_id]
                    self._save_local()
                return True
            except Exception as e:
                logger.warning(f"[NodeStore] DB error in delete(), falling back to local: {e}")

        self._ensure_local_loaded()
        if node_id in self._local_nodes:
            del self._local_nodes[node_id]
            self._save_local()
            logger.info(f"[NodeStore] Deleted local node: {node_id}")
            return True
        return False

    def get_global_nodes(self) -> List[Dict]:
        if cfg.USE_DB:
            try:
                rows = execute_query("SELECT * FROM nodes WHERE is_global = TRUE")
                if rows:
                    return [self._format(r) for r in rows]
            except Exception as e:
                logger.warning(f"[NodeStore] DB error in get_global_nodes(): {e}")

        self._ensure_local_loaded()
        return [n for n in self._local_nodes.values() if n.get("is_global")]

    def _format(self, row: Dict) -> Dict:
        """Format DB row to match expected dict structure."""
        if not row:
            return None
        return {
            "node_id": row["node_id"],
            "name": row["name"],
            "type": row["type"],
            "prompt": row["prompt"],
            "static_message": row.get("static_message"),
            "variables_needed": row.get("variables_needed", []),
            "tools": row.get("tools", []),
            "is_global": row.get("is_global", False),
            "max_turns": row.get("max_turns", 3),
            "transitions": row.get("transitions", []),
            "created_at": str(row.get("created_at", "")),
        }


# Initialize store
node_store = NodeStore()
