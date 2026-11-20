from __future__ import annotations
import json
import os
import time
from vllm_backend_full import chat_vllm
from prompts import CONSOLIDATION_REPAIR, CONSOLIDATION_REPAIR_ARGS


def obj(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


class Model:
    def __init__(
        self, name="qwen3.5:9b", url=None, num_ctx=32768, num_predict=1800, stream=True
    ):
        self.settings = dict(
            name=name,
            url=(url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")),
            num_ctx=num_ctx,
            num_predict=num_predict,
            stream=stream,
        )
        self.calls = []

    def ask(self, system, data, name, schema, validate):
        base_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ]
        messages = list(base_messages)

        budgets = [
            int(self.settings["num_predict"]),
            max(int(self.settings["num_predict"]), 2400),
            max(int(self.settings["num_predict"]), 3000),
            max(int(self.settings["num_predict"]), 3600),
        ]

        for attempt, budget in enumerate(budgets):
            start = time.monotonic()
            repair = attempt > 0
            result = chat_vllm(
                messages,
                self.settings["name"],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": "Submit the requested consolidation result.",
                            "parameters": schema,
                        },
                    }
                ],
                stream=self.settings["stream"],
                temperature=0.0 if repair else 0.3,
                max_tokens=budget,
                reasoning_effort="none" if repair else "low",
                enable_thinking=not repair,
                timeout=900.0,
                base_url=self.settings["url"],
                print_reasoning=not repair,
            )
            record = {
                "seconds": time.monotonic() - start,
                "content": result.get("content") or "",
                "tool_calls": result.get("tool_calls") or [],
                "done_reason": result.get("finish_reason"),
                "eval_count": result.get("completion_tokens"),
                "prompt_eval_count": result.get("prompt_tokens"),
                "attempt": attempt + 1,
                "max_tokens": budget,
                "repair": repair,
            }
            self.calls.append(record)
            args = None
            raw_args = None
            try:
                calls = record["tool_calls"]
                if (
                    len(calls) != 1
                    or (calls[0].get("function") or {}).get("name") != name
                ):
                    raise ValueError(f"Call {name} exactly once.")
                args = (calls[0].get("function") or {}).get("arguments")
                raw_args = args
                if isinstance(args, str):
                    text = args.strip()

                    if text.startswith("```"):
                        lines = text.splitlines()
                        if lines and lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        text = "\n".join(lines).strip()
                    args = json.loads(text)
                accepted = validate(args)
                record["accepted"] = True
                return accepted
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                record.update(accepted=False, error=str(error))
                if attempt == len(budgets) - 1:
                    raise ValueError(
                        f"Submission failed after {len(budgets)-1} repair attempts: {error}"
                    ) from error

                bad = ""
                if isinstance(raw_args, str) and raw_args.strip():

                    bad = raw_args.strip()
                    if len(bad) > 8000:
                        bad = bad[:4000] + "\n...[truncated]...\n" + bad[-4000:]
                messages = base_messages + [
                    {
                        "role": "user",
                        "content": (
                            CONSOLIDATION_REPAIR.format(name=name, error=error)
                            + (CONSOLIDATION_REPAIR_ARGS.format(bad=bad) if bad else "")
                        ),
                    }
                ]
