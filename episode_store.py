from __future__ import annotations

import json
import fcntl
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INDEX_FILE = "episode_index.json"
EPISODES_DIR = "episodes"
SCHEMA_VERSION = 2


def security5_scenario_vector(initial_state: dict) -> list[float]:
    security5 = initial_state.get("security5") or {}
    raw = security5.get("vector")

    if not isinstance(raw, (list, tuple)) or len(raw) != 5:
        raise ValueError(
            "Initial state must contain a five-component Security5 vector."
        )

    if any(isinstance(x, bool) for x in raw):
        raise ValueError("Security5 components cannot be booleans.")

    vector = [float(x) for x in raw]

    if not all(math.isfinite(x) and x >= 0.0 for x in vector):
        raise ValueError("Security5 components must be finite and nonnegative.")

    return vector


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _index_path(memory_dir: Path) -> Path:
    return Path(memory_dir) / INDEX_FILE


def _episodes_dir(memory_dir: Path) -> Path:
    return Path(memory_dir) / EPISODES_DIR


def _episode_path(memory_dir: Path, episode_id: str) -> Path:
    return _episodes_dir(memory_dir) / f"{episode_id}.json"


def _ensure_dirs(memory_dir: Path) -> None:
    Path(memory_dir).mkdir(parents=True, exist_ok=True)
    _episodes_dir(memory_dir).mkdir(parents=True, exist_ok=True)


def _empty_index() -> dict:
    return {"schema_version": SCHEMA_VERSION, "episodes": []}


def load_index(memory_dir: Path) -> dict:
    p = _index_path(memory_dir)
    if not p.exists():
        return _empty_index()
    try:
        data = json.loads(p.read_text())
    except Exception:
        return _empty_index()
    if not isinstance(data, dict) or not isinstance(data.get("episodes"), list):
        return _empty_index()

    if int(data.get("schema_version", -1)) != SCHEMA_VERSION:
        return _empty_index()
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False))
    tmp.replace(path)


def _save_index(memory_dir: Path, index: dict) -> None:
    _write_json(_index_path(memory_dir), index)


def _next_episode_id(index: dict) -> str:

    return "episode_" + uuid.uuid4().hex


def _update_index_entry(memory_dir: Path, episode: dict) -> None:
    _ensure_dirs(memory_dir)
    lock_path = Path(memory_dir) / ".episode_index.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = load_index(memory_dir)
        entry = {
            "episode_id": episode["episode_id"],
            "file": f"{EPISODES_DIR}/{episode['episode_id']}.json",
            "profile": episode.get("profile"),
            "status": episode.get("status"),
            "solved": bool(episode.get("final", {}).get("solved", False)),
            "scenario_representation": episode.get("scenario", {}).get(
                "representation"
            ),
            "scenario_vector": episode.get("scenario", {}).get("vector", []),
            "test_plan_calls": len(episode.get("steps", [])),
            "n_kept_plan_updates": episode.get("final", {}).get(
                "n_kept_plan_updates", 0
            ),
            "total_cost_musd": episode.get("final", {}).get("total_cost_musd"),
            "model": episode.get("metadata", {}).get("model"),
            "actor": episode.get("metadata", {}).get("actor"),
            "created_at": episode.get("created_at"),
            "updated_at": episode.get("updated_at"),
        }
        for i, old in enumerate(index["episodes"]):
            if old.get("episode_id") == episode["episode_id"]:
                index["episodes"][i] = entry
                break
        else:
            index["episodes"].append(entry)
        _save_index(memory_dir, index)
        fcntl.flock(lock, fcntl.LOCK_UN)


def start_episode(
    memory_dir: Path,
    profile: str,
    scenario_vector: list[float],
    initial_state: dict,
    *,
    network: str = "",
    metadata: dict | None = None,
    scenario_representation: str = "scenario_vector",
) -> str:
    _ensure_dirs(memory_dir)
    index = load_index(memory_dir)
    episode_id = _next_episode_id(index)
    timestamp = _now()

    episode = {
        "episode_id": episode_id,
        "profile": profile,
        "status": "in_progress",
        "created_at": timestamp,
        "updated_at": timestamp,
        "network": network,
        "metadata": dict(metadata or {}),
        "scenario": {
            "representation": str(scenario_representation),
            "vector": [float(x) for x in scenario_vector],
        },
        "initial_state": initial_state,
        "steps": [],
        "memory_events": [],
        "final": {
            "solved": False,
            "final_plan": [],
            "total_cost_musd": 0.0,
            "n_test_plan_calls": 0,
            "n_kept_plan_updates": 0,
            "final_state": initial_state,
        },
    }
    _write_json(_episode_path(memory_dir, episode_id), episode)
    _update_index_entry(memory_dir, episode)
    return episode_id


def load_episode(memory_dir: Path, episode_id: str) -> dict:
    p = _episode_path(memory_dir, episode_id)
    if not p.exists():
        raise FileNotFoundError(f"episode not found: {episode_id}")
    return json.loads(p.read_text())


def append_step(
    memory_dir: Path,
    episode_id: str,
    *,
    state_before: dict,
    attempted_actions: list[dict],
    attempted_state: dict,
    consequence: dict,
    costs: dict,
    search_decision: str,
    outcome_label: str,
    normalization_notes: list[str] | None = None,
    no_effect_or_invalid_action_notes: list[str] | None = None,
) -> int:
    episode = load_episode(memory_dir, episode_id)
    step_number = len(episode.get("steps", [])) + 1
    batch_size = len(attempted_actions)

    step = {
        "test_plan_call": step_number,
        "state_before": state_before,
        "attempted_actions": attempted_actions,
        "batch_size": batch_size,
        "attribution": (
            "none"
            if batch_size == 0
            else "individual" if batch_size == 1 else "batch_level"
        ),
        "simulator_consequence": {
            "attempted_state": attempted_state,
            "change": consequence,
        },
        "cost": costs,
        "search_decision": search_decision,
        "outcome_label": outcome_label,
        "normalization_notes": list(normalization_notes or []),
        "no_effect_or_invalid_action_notes": list(
            no_effect_or_invalid_action_notes or []
        ),
        "recorded_at": _now(),
    }

    episode.setdefault("steps", []).append(step)
    episode["updated_at"] = _now()
    episode["final"]["n_test_plan_calls"] = len(episode["steps"])

    _write_json(_episode_path(memory_dir, episode_id), episode)
    _update_index_entry(memory_dir, episode)
    return step_number


def append_memory_event(
    memory_dir: Path,
    episode_id: str,
    event: dict,
) -> int:
    episode = load_episode(memory_dir, episode_id)

    events = episode.setdefault("memory_events", [])

    record = dict(event)
    record["recorded_at"] = _now()

    events.append(record)
    episode["updated_at"] = _now()

    _write_json(
        _episode_path(memory_dir, episode_id),
        episode,
    )
    _update_index_entry(memory_dir, episode)

    return len(events)


def finish_episode(
    memory_dir: Path,
    episode_id: str,
    *,
    solved: bool,
    final_plan: list[dict],
    total_cost_musd: float,
    final_state: dict,
    n_kept_plan_updates: int = 0,
    status: str | None = None,
) -> None:
    episode = load_episode(memory_dir, episode_id)
    if status is None:
        status = "solved" if solved else "stopped"

    episode["status"] = status
    episode["updated_at"] = _now()
    episode["final"] = {
        "solved": bool(solved),
        "final_plan": list(final_plan),
        "total_cost_musd": round(float(total_cost_musd), 4),
        "n_test_plan_calls": len(episode.get("steps", [])),
        "n_kept_plan_updates": int(n_kept_plan_updates),
        "final_state": final_state,
    }
    _write_json(_episode_path(memory_dir, episode_id), episode)
    _update_index_entry(memory_dir, episode)
