from __future__ import annotations

from .episodes import recall_experience
from .tools import (
    analyse,
    commit_candidate,
    scan_day,
    test_staged_candidate,
    test_crossover_plan,
)

_ACTION_PROPERTIES = {
    "action": {
        "type": "string",
        "enum": [
            "upgrade_line",
            "upgrade_transformer",
            "new_line",
            "set_bus_reactive_support",
        ],
    },
    "line": {"type": "string"},
    "transformer": {"type": "string"},
    "target_capacity_mva": {
        "type": "number",
        "description": "Final total rating for upgrade_line.",
    },
    "target_sn_mva": {
        "type": "number",
        "description": "Final total rating for upgrade_transformer.",
    },
    "from_bus": {"type": "integer"},
    "to_bus": {"type": "integer"},
    "capacity_mva": {
        "type": "number",
        "description": "Capacity of a planned new line.",
    },
    "bus": {"type": "string"},
    "target_mvar": {
        "type": "number",
        "description": (
            "Final fixed reactive-support target at the bus. Positive raises "
            "voltage (capacitive), negative lowers voltage (inductive), and 0 "
            "removes planned support."
        ),
    },
}

_ACTION_ITEM = {
    "type": "object",
    "properties": _ACTION_PROPERTIES,
    "required": ["action"],
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scan_day",
            "description": (
                "Inspect the current scenario over all 24 hours before planning. "
                "Returns the 24-hour load pattern, peak hour, and base-case stress/problem "
                "hours. This tool is descriptive only and does not change the grid or plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {"profile": {"type": "string"}},
                "required": ["profile"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyse",
            "description": (
                "Diagnose current whole-day base-case thermal and voltage problems and "
                "peak-hour N-1 contingency problems. Returns the physical elements and "
                "buses involved, severity information, and failed N-1 outage-test count. "
                "This tool is descriptive only and does not recommend or apply actions."
            ),
            "parameters": {
                "type": "object",
                "properties": {"profile": {"type": "string"}},
                "required": ["profile"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_experience",
            "description": (
                "Retrieve one completed historical planning episode using the episodic "
                "retrieval rule configured for the current run. The returned experience "
                "is historical evidence only and does not alter the current grid or retained plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {"type": "string"},
                    "rank": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Similarity rank; rank=1 is the highest-ranked available experience.",
                    },
                },
                "required": ["profile"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_candidate",
            "description": (
                "Commit ONE previously verified improving candidate when the verifier "
                "returned selection_required=true. Use this only after reasoning over the sibling "
                "AC/N-1 outcomes, side effects, and cost. Reasoning text does not apply a plan; "
                "this tool call is the explicit commit operation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {"type": "string"},
                    "candidate_id": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "One candidate ID listed in eligible_candidate_ids.",
                    },
                },
                "required": ["profile", "candidate_id"],
            },
        },
    },
]

STAGED_CANDIDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "test_staged_candidate",
        "description": (
            "Verify ONE standalone candidate plan against the retained parent, "
            "including every independent single-action AC PF and the full-plan AC PF. "
            "A, B, C are requested in separate isolated model calls; only D sees all evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "profile": {"type": "string"},
                "candidate_id": {"type": "string", "enum": ["A", "B", "C"]},
                "strategy": {"type": "string"},
                "actions": {"type": "array", "minItems": 1, "items": _ACTION_ITEM},
            },
            "required": ["profile", "candidate_id", "actions"],
        },
    },
}

TOOLS.append(STAGED_CANDIDATE_TOOL)

TOOLS.append(
    {
        "type": "function",
        "function": {
            "name": "test_crossover_plan",
            "description": (
                "Submit one complete supplied legal D option for independent verification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profile": {"type": "string"},
                    "round_id": {"type": "string"},
                    "action_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                    },
                    "strategy": {
                        "type": "string",
                        "description": (
                            "Copy the strategy field from the chosen legal option."
                        ),
                    },
                },
                "required": ["profile", "round_id", "action_refs", "strategy"],
            },
        },
    }
)


REGISTRY = {
    "scan_day": scan_day,
    "analyse": analyse,
    "test_staged_candidate": test_staged_candidate,
    "test_crossover_plan": test_crossover_plan,
    "commit_candidate": commit_candidate,
    "recall_experience": recall_experience,
}


def call_tool(name: str, args: dict) -> dict:
    from .tools.test_plan import pending_round_tool

    required = pending_round_tool()
    if required and name != required:
        return {
            "error": (
                f"The pending round requires {required} " "before any other tool."
            )
        }

    fn = REGISTRY.get(name)
    if fn is None:
        return {"error": f"no such tool '{name}'. Available: {list(REGISTRY)}"}
    allowed = {
        "scan_day": ("profile",),
        "analyse": ("profile",),
        "test_staged_candidate": ("profile", "candidate_id", "strategy", "actions"),
        "test_crossover_plan": ("profile", "round_id", "action_refs", "strategy"),
        "commit_candidate": ("profile", "candidate_id"),
        "recall_experience": ("profile", "rank"),
    }[name]
    try:
        return fn(**{key: value for key, value in args.items() if key in allowed})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
