from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict, deque
from typing import Any


def _normalise_function_call(call: dict, fallback_id: str) -> tuple[dict, str]:
    fn = call.get("function") or {}
    name = str(fn.get("name") or "")

    arguments = fn.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, separators=(",", ":"))

    call_id = str(call.get("id") or fallback_id)

    return (
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        },
        name,
    )


def _normalise_messages(messages: list[dict]) -> list[dict]:
    non_system: list[dict] = []
    system_parts: list[str] = []
    pending_ids: dict[str, deque[str]] = defaultdict(deque)

    for message_index, original in enumerate(messages):
        if not isinstance(original, dict):
            continue

        role = original.get("role")
        content = original.get("content")

        if role == "system":
            text = "" if content is None else str(content)
            if text.strip():
                system_parts.append(text.strip())
            continue

        if role == "user":
            non_system.append(
                {
                    "role": "user",
                    "content": "" if content is None else content,
                }
            )
            continue

        if role == "assistant":
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "" if content is None else content,
            }

            tool_calls = original.get("tool_calls") or []
            if tool_calls:
                fixed_calls = []
                for call_index, call in enumerate(tool_calls):
                    if not isinstance(call, dict):
                        continue
                    fixed, name = _normalise_function_call(
                        call,
                        fallback_id=f"call_{message_index}_{call_index}",
                    )
                    fixed_calls.append(fixed)
                    if name:
                        pending_ids[name].append(fixed["id"])

                if fixed_calls:
                    msg["tool_calls"] = fixed_calls

            non_system.append(msg)
            continue

        if role == "tool":
            name = str(original.get("name") or "")
            explicit_id = original.get("tool_call_id")

            if explicit_id:
                tool_call_id = str(explicit_id)
            elif name and pending_ids[name]:
                tool_call_id = pending_ids[name].popleft()
            else:
                raise ValueError(
                    "Cannot map tool-result message to a preceding assistant "
                    f"tool call: tool={name!r}, message_index={message_index}."
                )

            non_system.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "" if content is None else content,
                }
            )
            continue

    out: list[dict] = []
    if system_parts:
        out.append(
            {
                "role": "system",
                "content": "\n\n".join(system_parts),
            }
        )
    out.extend(non_system)
    return out


def _read_reasoning(message: dict) -> str:

    return str(message.get("reasoning") or message.get("reasoning_content") or "")


def _normalise_returned_tool_call(call: dict, fallback_id: str) -> dict:
    fn = call.get("function") or {}
    arguments = fn.get("arguments") or "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)

    return {
        "id": call.get("id") or fallback_id,
        "type": call.get("type") or "function",
        "function": {
            "name": fn.get("name") or "",
            "arguments": arguments,
        },
    }


def _merge_streamed_tool_call(states: dict[int, dict], delta_call: dict) -> None:
    index = int(delta_call.get("index") or 0)
    state = states.setdefault(
        index,
        {
            "id": None,
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )

    if delta_call.get("id"):
        state["id"] = delta_call["id"]

    if delta_call.get("type"):
        state["type"] = delta_call["type"]

    fn = delta_call.get("function") or {}

    if fn.get("name"):

        state["function"]["name"] = fn["name"]

    if fn.get("arguments"):
        state["function"]["arguments"] += fn["arguments"]


def _request(
    endpoint: str,
    payload: dict,
    api_key: str,
    timeout: float,
):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def chat_vllm(
    messages: list[dict],
    model: str,
    *,
    tools: list[dict] | None,
    stream: bool = True,
    temperature: float = 0.3,
    max_tokens: int = 3600,
    reasoning_effort: str | None = "low",
    enable_thinking: bool = True,
    top_p: float | None = None,
    top_k: int | None = None,
    timeout: float = 900.0,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "EMPTY",
    print_reasoning: bool = True,
    dim: str = "\033[2m",
    reset: str = "\033[0m",
) -> dict:
    """Execute one vLLM OpenAI-compatible chat-completion request."""
    endpoint = base_url.rstrip("/") + "/chat/completions"
    request_messages = _normalise_messages(messages)

    payload: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        "stream": bool(stream),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "chat_template_kwargs": {
            "enable_thinking": bool(enable_thinking),
        },
    }

    if enable_thinking:
        if reasoning_effort:
            payload["reasoning_effort"] = str(reasoning_effort)
    else:
        payload["reasoning_effort"] = "none"

    if top_p is not None:
        payload["top_p"] = float(top_p)
    if top_k is not None:
        payload["top_k"] = int(top_k)

    if tools:
        payload["tools"] = tools

        if len(tools) == 1:
            fn = (tools[0].get("function") or {}) if isinstance(tools[0], dict) else {}
            tool_name = fn.get("name")
            if tool_name:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_name},
                }
            else:
                payload["tool_choice"] = "auto"
        else:
            payload["tool_choice"] = "auto"

    if stream:
        payload["stream_options"] = {"include_usage": True}

    thinking_parts: list[str] = []
    content_parts: list[str] = []
    final_calls: list[dict] = []
    finish_reason = None
    prompt_tokens = None
    completion_tokens = None

    try:
        with _request(endpoint, payload, api_key, timeout) as response:
            if not stream:
                packet = json.loads(response.read().decode("utf-8"))
                choices = packet.get("choices") or []
                if not choices:
                    raise RuntimeError(f"vLLM returned no completion choices: {packet}")

                choice = choices[0]
                message = choice.get("message") or {}

                reasoning = _read_reasoning(message)
                if reasoning:
                    thinking_parts.append(reasoning)
                    if print_reasoning:
                        print(dim + reasoning.strip() + reset, flush=True)

                content = message.get("content") or ""
                if content:
                    content_parts.append(content)

                for index, call in enumerate(message.get("tool_calls") or []):
                    final_calls.append(
                        _normalise_returned_tool_call(
                            call,
                            fallback_id=f"call_{index}",
                        )
                    )

                finish_reason = choice.get("finish_reason")
                usage = packet.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")

            else:
                states: dict[int, dict] = {}
                printing = False

                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    if not data:
                        continue

                    packet = json.loads(data)

                    usage = packet.get("usage") or {}
                    if usage:
                        if usage.get("prompt_tokens") is not None:
                            prompt_tokens = usage["prompt_tokens"]
                        if usage.get("completion_tokens") is not None:
                            completion_tokens = usage["completion_tokens"]

                    choices = packet.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    reasoning = _read_reasoning(delta)
                    if reasoning:
                        thinking_parts.append(reasoning)
                        if print_reasoning:
                            if not printing:
                                sys.stdout.write(dim)
                                printing = True
                            sys.stdout.write(reasoning)
                            sys.stdout.flush()

                    content = delta.get("content") or ""
                    if content:
                        content_parts.append(content)

                    for call in delta.get("tool_calls") or []:
                        _merge_streamed_tool_call(states, call)

                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]

                if printing and print_reasoning:
                    sys.stdout.write(reset + "\n")
                    sys.stdout.flush()

                for index in sorted(states):
                    call = states[index]
                    call["id"] = call["id"] or f"call_{index}"
                    call["function"]["arguments"] = (
                        call["function"]["arguments"] or "{}"
                    )
                    final_calls.append(call)

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code} from {endpoint}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach vLLM at {endpoint}: {exc}") from exc

    return {
        "thinking": "".join(thinking_parts).strip(),
        "content": "".join(content_parts).strip(),
        "tool_calls": final_calls,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
