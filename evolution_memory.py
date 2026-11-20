from __future__ import annotations
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

VERSION = "trajectory-memory-v1"
ACTIVE = None


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha(value):
    return hashlib.sha256(dump(value).encode()).hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _security5_vector(state_or_security5):
    if not isinstance(state_or_security5, dict):
        return None
    sec = (
        state_or_security5.get("security5")
        if "security5" in state_or_security5
        else state_or_security5
    )
    sec = sec or {}
    values = sec.get("vector")
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        return None
    try:
        out = [float(x) for x in values]
    except (TypeError, ValueError):
        return None
    if any((not math.isfinite(x)) or x < 0 for x in out):
        return None
    return out


def vector(state):
    values = _security5_vector(state)
    if values is None:
        raise ValueError("Missing or invalid initial Security5.")
    return values


def event_tags(name, result):
    if result.get("error"):
        return {"tool_error"}
    tags = set()
    if name == "analyse":
        tags.add("planning_start")
    if result.get("crossover_required"):
        tags.add("crossover_pending")
    elif result.get("selection_required"):
        tags.add("selection_pending")
    committed = bool(result.get("selected_candidate_committed")) or (
        name == "commit_candidate" and not result.get("error")
    )
    if committed:
        tags.update(("after_commit", "candidate_generation"))
    if result.get("search_decision") == "reverted":
        tags.update(("nonimproving_round", "candidate_generation"))
        candidates = result.get("candidate_results") or []
        if any(
            c.get("security5_relation_vs_parent") == "security5_tradeoff"
            and not c.get("committed")
            for c in candidates
        ):
            tags.add("reverted_mixed")
    if not (tags & {"crossover_pending", "selection_pending"}) and name in (
        "analyse",
        "test_staged_candidate",
        "test_crossover_plan",
        "commit_candidate",
    ):
        tags.add("candidate_generation")
    return tags


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        if not self.file.closed:
            try:
                self.file.write(text)
                self.file.flush()
            except ValueError:
                pass
        return self.terminal.write(text)

    def flush(self):
        if not self.file.closed:
            try:
                self.file.flush()
            except ValueError:
                pass
        try:
            self.terminal.flush()
        except ValueError:
            pass

    def isatty(self):
        return self.terminal.isatty()

    def __getattr__(self, key):
        return getattr(self.terminal, key)


class PlanningSession:
    def __init__(
        self,
        G,
        profile,
        model,
        episode_memory=None,
    ):
        from grid_tools import config

        self.G, self.profile, self.model = G, profile, model
        self.memory = Path(config.MEMORY).resolve()
        self.root = self.memory / "evolution"
        self.episode_memory = (
            Path(episode_memory).resolve() if episode_memory else self.memory
        )
        self.network = config.TOPOLOGY
        self.mode = "episodes"
        self.bank = _bank()
        self.run_id = "run_" + uuid.uuid4().hex
        self.path = self.root / "runs" / self.run_id
        self.path.mkdir(parents=True)
        self.manifest = {
            "version": VERSION,
            "run_id": self.run_id,
            "profile": profile,
            "network": self.network,
            "model": model,
            "mode": self.mode,
            "started_at": now(),
            "status": "in_progress",
            "episode_id": None,
            "episode_memory": str(self.episode_memory),
            "strategy_bank_version": self.bank.get("bank_version", 1),
            "strategy_bank_sha256": sha(self.bank),
        }
        self.tags = set()
        self.generation = 0
        self.current_generation = None
        self.chat_starts = {}
        self.model_seconds = 0.0
        self.wall_start = time.monotonic()

    def __enter__(self):
        global ACTIVE
        if ACTIVE:
            raise RuntimeError("A planning recording session is already active.")
        self._old_episode_read = os.environ.get("EPISODE_MEMORY_READ")
        os.environ["EPISODE_MEMORY_READ"] = str(self.episode_memory)
        self.log = (self.path / "console.log").open("w", encoding="utf-8")
        self.events = (self.path / "events.jsonl").open("w", encoding="utf-8")
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(self.stdout, self.log)
        sys.stderr = Tee(self.stderr, self.log)
        ACTIVE = self
        save(self.path / "manifest.json", self.manifest)
        save(self.path / "strategy_bank_used.json", self.bank)
        print(
            f"memory: mode={self.mode} | strategy bank v{self.bank.get('bank_version', 1)} | "
            f"episodic source={self.episode_memory} | run={self.run_id}",
            flush=True,
        )
        return self

    def record(self, kind, **data):
        self.events.write(dump({"kind": kind, "time": now(), **data}) + "\n")
        self.events.flush()

    def tool(self, name, args, result):
        self.record(
            "tool_result",
            generation_id=self.current_generation,
            name=name,
            args=args,
            result=result,
        )
        if result.get("episode_id"):
            self.manifest["episode_id"] = result["episode_id"]
        self.tags = event_tags(name, result)
        if name == "recall_experience" and not result.get("error"):
            self.record("episode_retrieval_explicit", result=copy.deepcopy(result))

    def __exit__(self, exc_type, exc, tb):
        global ACTIVE
        try:
            if exc_type:
                traceback.print_exception(exc_type, exc, tb)
            eid = self.manifest.get("episode_id")
            if not eid:
                state = importlib.import_module(ACTIVE.G.__name__ + ".state")

                eid = getattr(state, "ACTIVE_EPISODE_ID", None)
            self.manifest["episode_id"] = eid
            if eid:
                ep_path = self.memory / "episodes" / (str(eid) + ".json")
                if ep_path.exists():
                    ep = read(ep_path)
                    save(self.path / "episode.json", ep)
                    self.manifest["episode_sha256"] = sha(ep)
                    self.manifest["episode_status"] = ep.get("status")
            complete = not exc_type and self.manifest.get("episode_status") in (
                "solved",
                "stopped",
                "failed",
                "unsolved",
            )
            self.manifest.update(
                status=(
                    "completed"
                    if complete
                    else (
                        "interrupted" if exc_type is KeyboardInterrupt else "incomplete"
                    )
                ),
                ended_at=now(),
                error=str(exc) if exc else None,
                model_seconds=self.model_seconds,
                elapsed_seconds=time.monotonic() - self.wall_start,
            )
            self.record("run_end", manifest=self.manifest)
            print(f"memory: log saved at {self.path/'console.log'}", flush=True)
        finally:
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            ACTIVE = None
            self.events.close()
            self.log.close()
            if self._old_episode_read is None:
                os.environ.pop("EPISODE_MEMORY_READ", None)
            else:
                os.environ["EPISODE_MEMORY_READ"] = self._old_episode_read
            self.manifest["log_sha256"] = hashlib.sha256(
                (self.path / "console.log").read_bytes()
            ).hexdigest()
            self.manifest["events_sha256"] = hashlib.sha256(
                (self.path / "events.jsonl").read_bytes()
            ).hexdigest()
            save(self.path / "manifest.json", self.manifest)
        return False


def before_chat(payload):
    if not ACTIVE:
        return None
    ACTIVE.generation += 1
    gid = f"g{ACTIVE.generation}"
    ACTIVE.current_generation = gid
    ACTIVE.chat_starts[gid] = time.monotonic()
    ACTIVE.record(
        "model_input",
        generation_id=gid,
        payload=payload,
        tags=sorted(ACTIVE.tags),
    )
    return gid


def after_chat(gid, result):
    if ACTIVE:
        elapsed = time.monotonic() - ACTIVE.chat_starts.pop(gid, time.monotonic())
        ACTIVE.model_seconds += elapsed
        ACTIVE.record("model_output", generation_id=gid, result=result, seconds=elapsed)
        if not result.get("tool_calls"):
            ACTIVE.tags.add("no_tool_call")


def record_tool(name, args, result):
    if ACTIVE:
        ACTIVE.tool(name, args, result)


def set_episode_metadata():
    if ACTIVE:
        state = importlib.import_module(ACTIVE.G.__name__ + ".state")

        state.RUN_CONTEXT["evolution"] = {
            "run_id": ACTIVE.run_id,
            "mode": ACTIVE.mode,
            "strategy_bank_version": ACTIVE.bank.get("bank_version", 1),
            "strategy_bank_sha256": sha(ACTIVE.bank),
            "log_file": str(ACTIVE.path / "console.log"),
            "events_file": str(ACTIVE.path / "events.jsonl"),
            "episode_memory_read": str(ACTIVE.episode_memory),
        }


from prompts import (
    STRATEGY_INSTRUCTION,
    STRATEGY_SIZING_INSTRUCTION,
    STRATEGY_ROLE_TEXT,
    STRATEGY_CANDIDATE_A_NOTE,
)

STRATEGY_K = 3
_BANK_PATH = None
_BANK = None


def set_strategy_bank(path):
    """Use the strategy bank at path (None disables procedural memory)."""
    global _BANK_PATH, _BANK
    _BANK_PATH, _BANK = (str(path) if path else None), None


def _bank():
    global _BANK
    if _BANK is None and _BANK_PATH:
        _BANK = load_bank(_BANK_PATH)
    return _BANK


BASE_LIMIT, N1_LIMIT = 100.0, 110.0


V_MIN, V_MIN_N1 = 0.95, 0.92


def line_needs(state: dict) -> dict:
    out = {}
    for it in state.get("base_case_overloaded_elements") or []:
        if it.get("element_type", "line") != "line":
            continue
        pct = it.get("loading_pct", it.get("worst_loading_pct"))
        cap = it.get("rating_mva", it.get("capacity_mva"))
        if pct is None or cap is None:
            continue
        out[it["element"]] = {
            "rating": float(cap),
            "pct": float(pct),
            "kind": "base",
            "need": float(cap) * float(pct) / BASE_LIMIT,
        }
    for it in state.get("n1_overloaded_elements") or []:
        name = it.get("element")
        if not name or not str(name).startswith("Line_"):
            continue
        pct = it.get("worst_loading_pct", it.get("loading_pct"))
        cap = it.get("capacity_mva", it.get("rating_mva"))
        if pct is None or cap is None:
            continue
        need = float(cap) * float(pct) / N1_LIMIT
        if name not in out or need > out[name]["need"]:
            out[name] = {
                "rating": float(cap),
                "pct": float(pct),
                "kind": "n1",
                "need": need,
            }
    return out


def bus_depths(state: dict) -> dict:
    out = {}
    for it in state.get("base_case_voltage_problem_buses") or []:
        vm = it.get("min_vm_pu", it.get("worst_vm_pu"))
        if it.get("bus") and vm is not None and float(vm) < V_MIN:
            out[it["bus"]] = max(out.get(it["bus"], 0.0), V_MIN - float(vm))
    for it in state.get("n1_voltage_problem_buses") or []:
        vm = it.get("worst_vm_pu")
        if it.get("bus") and vm is not None:
            out[it["bus"]] = max(out.get(it["bus"], 0.0), V_MIN_N1 - float(vm), 0.001)
    return out


SCHEMA = "scope-generic-strategies-v1"


def extent_of(n_violated_lines):
    return (
        "few"
        if n_violated_lines <= 6
        else "several" if n_violated_lines <= 12 else "many"
    )


def severity_of(over_limit):
    return (
        "mild" if over_limit < 1.05 else "moderate" if over_limit <= 1.30 else "severe"
    )


def features(state, first_round):
    needs = line_needs(state)
    volt = bus_depths(state)
    comps = set()
    if any(n["kind"] == "n1" for n in needs.values()):
        comps.add("n1_thermal")
    if any(n["kind"] == "base" for n in needs.values()):
        comps.add("base_thermal")
    if volt:
        comps.add("voltage")
    worst = max([n["need"] / n["rating"] for n in needs.values()] or [1.0])
    sev = "mild" if worst < 1.05 else "moderate" if worst <= 1.30 else "severe"
    return {
        "stage": "first_plan" if first_round else "residual_repair",
        "violation": (
            next(iter(comps)) if len(comps) == 1 else ("mixed" if comps else "none")
        ),
        "components": sorted(comps),
        "severity": sev,
        "extent": extent_of(len(needs)),
        "n_overloaded_lines": len(needs),
    }


def select(bank, feat, k=3):
    hits = []
    for s in bank["strategies"]:
        t = s.get("trigger") or {}
        st, vi, se = (
            t.get("stage") or ["any"],
            t.get("violation") or ["any"],
            t.get("severity") or ["any"],
        )
        ok_stage = "any" in st or feat["stage"] in st
        ok_viol = (
            "any" in vi
            or feat["violation"] in vi
            or (
                feat["violation"] == "mixed" and bool(set(vi) & set(feat["components"]))
            )
        )
        ok_sev = "any" in se or feat["severity"] in se
        ex = t.get("extent") or ["any"]
        ok_ext = "any" in ex or feat.get("extent") in ex
        if ok_stage and ok_viol and ok_sev and ok_ext and feat["violation"] != "none":
            spec = sum(1 for d in (st, vi, se, ex) if "any" not in d)
            hits.append((len(s.get("supporting_episode_ids") or []), spec, s))

    hits.sort(key=lambda x: (-x[1], -x[0], x[2]["strategy_id"]))
    return [s for _, _, s in hits[:k]]


ROLE = {"B": "low", "A": "mid", "C": "high"}


def select_relaxed(bank, feat, k=1):
    feat_comps = set(feat.get("components") or [])
    sizing = [
        s
        for s in bank["strategies"]
        if (s.get("sizing_rule") or {}).get("applies_to") not in (None, "none")
    ]

    def viol_ok(s):
        vi = (s.get("trigger") or {}).get("violation") or ["any"]
        return (
            "any" in vi
            or feat["violation"] in vi
            or bool(set(vi) & feat_comps)
            or ("mixed" in vi and len(feat_comps) > 1)
        )

    def stage_ok(s):
        st = (s.get("trigger") or {}).get("stage") or ["any"]
        return "any" in st or feat["stage"] in st

    for cond in (lambda s: stage_ok(s) and viol_ok(s), viol_ok, lambda s: True):
        hits = [s for s in sizing if cond(s)]
        if hits:
            hits.sort(
                key=lambda s: (
                    -len(s.get("supporting_episode_ids") or []),
                    s["strategy_id"],
                )
            )
            return hits[:k]
    return []


def bind_for_candidate(bank, state, first_round, candidate_id, k=3):
    feat = features(state, first_round)
    chosen = select(bank, feat, k=k)
    applied = apply_sizing(chosen, state, candidate_id)
    fallback = False
    if not applied and line_needs(state):
        fb = select_relaxed(bank, feat)
        applied = apply_sizing(fb, state, candidate_id)
        if applied:
            fallback = True
            applied["matched_by"] = "fallback (no exact trigger match)"
            chosen = chosen + [x for x in fb if x not in chosen]
    return chosen, applied, fallback, feat


def apply_sizing(strategies, state, candidate_id="A"):
    needs = line_needs(state)
    role = ROLE.get(candidate_id, "mid")
    for s in strategies:
        r = s.get("sizing_rule") or {}
        if r.get("applies_to") in (None, "none"):
            continue
        rows = []
        for line, n in sorted(
            needs.items(), key=lambda kv: -kv[1]["need"] / kv[1]["rating"]
        ):
            kind = "n1_bound" if n["kind"] == "n1" else "base_bound"
            if r["applies_to"] not in ("any", kind):
                continue
            sv = severity_of(n["need"] / n["rating"])
            m = r["margins"][sv]
            pick = {
                "low": m["low"],
                "mid": (m["low"] + m["high"]) / 2,
                "high": m["high"],
            }[role]
            rows.append(
                {
                    "line": line,
                    "current_worst_loading_pct": round(n["pct"], 1),
                    "binding_limit": "N-1 110%" if n["kind"] == "n1" else "base 100%",
                    "overload_severity": sv,
                    "current_rating_mva": n["rating"],
                    "rating_needed_for_current_flow_mva": round(n["need"], 1),
                    "target_range_mva": [
                        math.ceil(n["need"] * m["low"]),
                        math.ceil(n["need"] * m["high"]),
                    ],
                    "target_mva_for_this_candidate": math.ceil(n["need"] * pick),
                }
            )
        if rows:
            return {
                "from_strategy": s["strategy_id"],
                "candidate_role": role,
                "lines": rows,
            }
    return None


def for_packet(s):
    return {
        k: s[k]
        for k in (
            "strategy_id",
            "name",
            "applies_when",
            "procedure",
            "avoid",
            "watch_for",
            "sizing_rule",
        )
    }


def load_bank(path):
    b = json.load(open(path))
    if b.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} file: {path}")
    return b


def procedural_packet(candidate_id, current_state):
    bank = _bank()
    if bank is None or not isinstance(current_state, dict) or ACTIVE is None:
        return {}
    if ACTIVE.profile in bank.get("profiles", []):
        raise RuntimeError(f"LEAKAGE: {ACTIVE.profile} is inside its own strategy bank")
    first_round = not (current_state.get("current_plan") or [])
    chosen, applied, fallback, feat = bind_for_candidate(
        bank, current_state, first_round, candidate_id, k=STRATEGY_K
    )
    out = {}
    if chosen:
        out["scope2_strategies"] = [for_packet(x) for x in chosen]
        if applied:
            out["scope2_strategy_applied_to_current_diagnostics"] = applied
        instr = STRATEGY_INSTRUCTION + (
            STRATEGY_SIZING_INSTRUCTION.format(
                role=STRATEGY_ROLE_TEXT[applied["candidate_role"]]
            )
            if applied
            else ""
        )
        if candidate_id == "A":
            instr += STRATEGY_CANDIDATE_A_NOTE
        out["scope2_instruction"] = instr
    ACTIVE.record(
        "scope2_strategies",
        candidate_id=candidate_id,
        strategy_ids=[x["strategy_id"] for x in chosen],
        features=feat,
        sizing_bound_lines=len((applied or {}).get("lines") or []),
        sizing_fallback=fallback,
    )
    return out
