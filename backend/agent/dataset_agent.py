"""
VisCurator / CVAgent — Dataset Curation Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ReAct agent powered by NVIDIA NIM (Llama-3.1-70B) using the OpenAI
**native function-calling** API.  The agent:

1. Receives a natural-language curation request.
2. Calls tools via the model's ``tool_calls`` mechanism (no fragile
   JSON-in-text parsing).
3. Executes each tool and feeds the result back as a ``tool`` role message.
4. When the model calls ``generate_and_run_script``, the subprocess stdout
   is streamed line-by-line to the frontend terminal via ``SCRIPT_LOG``
   messages.
5. Terminates when the model stops issuing tool calls and returns a plain
   text final answer, or after ``MAX_ITERATIONS`` rounds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine
from uuid import UUID

from backend.agent.nim_client import NIMClient
from backend.agent.tools import (
    TOOL_REGISTRY,
    execute_tool,
    set_script_emit_callback,
)
from backend.models.schemas import MessageType, PipelineMessage

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10

EmitFn = Callable[[PipelineMessage], Coroutine[Any, Any, None]]

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are **CVAgent**, an expert AI agent for computer-vision dataset curation
built into VisCurator.

Your job when given a dataset request:
1. Call `search_huggingface` to find relevant datasets.
2. Call `get_dataset_info` on the most promising result to inspect its splits,
   features, and size.
3. Optionally call `estimate_dataset_quality` to score image quality.
4. Call `generate_and_run_script` with the chosen dataset_id (and the correct
   label_column / image_column you discovered in step 2) to download, filter,
   and split the data.  Pass the user's target_size.
5. After the script finishes, return a concise final summary to the user.

Rules:
- Always reason step-by-step before each tool call.
- Never invent dataset IDs — only use IDs returned by the search tool.
- If a tool returns an error, try an alternative dataset or approach.
- Your final message (when you stop calling tools) must be a clear, structured
  summary: recommended dataset, quality notes, output location, and next steps.
"""

# ── Build the OpenAI-format tools list from TOOL_REGISTRY ────────────────────

def _nim_tools() -> list[dict[str, Any]]:
    """Convert TOOL_REGISTRY entries into the OpenAI function-calling schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": entry["name"],
                "description": entry["description"],
                "parameters": entry["parameters"],
            },
        }
        for entry in TOOL_REGISTRY.values()
    ]


# ── Agent ─────────────────────────────────────────────────────────────────────

class DatasetAgent:
    """Stateful agent that curates datasets via NIM native function calling.

    Parameters
    ----------
    nim_client:
        Initialised :class:`NIMClient`.
    job_id:
        Unique pipeline job identifier.
    emit:
        Async callback that sends a :class:`PipelineMessage` to the WebSocket.
    """

    def __init__(self, nim_client: NIMClient, job_id: UUID, emit: EmitFn) -> None:
        self._nim = nim_client
        self._job_id = job_id
        self._emit = emit
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]
        self._iteration = 0
        self._tool_results: list[dict[str, Any]] = []

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        source: str = "huggingface",
        target_size: int = 1000,
    ) -> dict[str, Any]:
        """Execute the full agentic curation pipeline and return a result dict."""

        user_msg = (
            f"I need a curated computer-vision dataset.\n"
            f"- Query / topic: {query}\n"
            f"- Preferred source: {source}\n"
            f"- Target size: {target_size} images\n\n"
            f"Search for relevant datasets, pick the best one, inspect it, "
            f"then generate and run the download script."
        )
        self._messages.append({"role": "user", "content": user_msg})

        await self._emit(PipelineMessage(
            type=MessageType.LOG,
            message=(
                f'Pipeline started — query="{query}"  '
                f"source={source}  target_size={target_size}"
            ),
            data={"query": query, "source": source, "target_size": target_size},
        ))

        # Register the script-output emit callback so generate_and_run_script
        # can stream subprocess lines directly to the terminal.
        async def _script_line_emit(line: str, stream: str = "stdout") -> None:
            await self._emit(PipelineMessage(
                type=MessageType.SCRIPT_LOG,
                message=line,
                data={"stream": stream},
            ))

        set_script_emit_callback(_script_line_emit)

        tools = _nim_tools()
        final_answer = ""

        # ── ReAct / function-calling loop ─────────────────────────────────────
        while self._iteration < MAX_ITERATIONS:
            self._iteration += 1

            await self._emit(PipelineMessage(
                type=MessageType.LOG,
                message=f"Thinking … (step {self._iteration}/{MAX_ITERATIONS})",
            ))

            # ── 1. Call the LLM ──
            try:
                response = await self._nim.chat(
                    messages=self._messages,
                    temperature=0.1,
                    max_tokens=2048,
                    stream=False,
                    tools=tools,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.exception("LLM call failed at iteration %d", self._iteration)
                await self._emit(PipelineMessage(
                    type=MessageType.ERROR,
                    message=f"LLM inference error: {exc}",
                ))
                break

            choice = response.choices[0]  # type: ignore[union-attr]
            msg = choice.message

            # Append the raw assistant message to the conversation
            assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            self._messages.append(assistant_entry)

            # ── 2. Stream the assistant's reasoning text (if any) ──
            if msg.content and msg.content.strip():
                await self._emit(PipelineMessage(
                    type=MessageType.THOUGHT,
                    message=msg.content.strip(),
                    data={"iteration": self._iteration},
                ))

            # ── 3. No tool calls → model is done ──
            if not msg.tool_calls:
                final_answer = msg.content or ""
                break

            # ── 4. Execute each tool call ──
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # Inject target_size into generate_and_run_script if not set
                if fn_name == "generate_and_run_script" and "target_size" not in fn_args:
                    fn_args["target_size"] = target_size

                await self._emit(PipelineMessage(
                    type=MessageType.TOOL_CALL,
                    message=f"Calling {fn_name}",
                    data={"tool": fn_name, "arguments": fn_args, "iteration": self._iteration},
                ))

                tool_result = await execute_tool(fn_name, fn_args)

                self._tool_results.append({
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result_summary": _truncate_result(tool_result),
                    "iteration": self._iteration,
                })

                await self._emit(PipelineMessage(
                    type=MessageType.TOOL_RESULT,
                    message=f"{fn_name} completed",
                    data=tool_result,
                ))

                # Feed the tool result back as a tool-role message
                result_text = json.dumps(tool_result, default=str)
                if len(result_text) > 6000:
                    result_text = result_text[:6000] + "\n... (truncated)"

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

        # ── Max iterations fallback ───────────────────────────────────────────
        if not final_answer and self._iteration >= MAX_ITERATIONS:
            await self._emit(PipelineMessage(
                type=MessageType.LOG,
                message="Max iterations reached — generating final summary …",
            ))
            self._messages.append({
                "role": "user",
                "content": (
                    "You have reached the maximum number of steps. "
                    "Please provide your best final summary now, without calling any more tools."
                ),
            })
            try:
                resp2 = await self._nim.chat(
                    messages=self._messages,
                    temperature=0.1,
                    max_tokens=1024,
                    stream=False,
                )
                final_answer = resp2.choices[0].message.content or ""  # type: ignore[union-attr]
            except Exception:
                final_answer = "Agent reached max iterations before completing."

        final_result: dict[str, Any] = {
            "summary": final_answer,
            "tool_calls_made": self._iteration,
            "tool_results": self._tool_results,
        }

        await self._emit(PipelineMessage(
            type=MessageType.DONE,
            message=final_answer,
            data=final_result,
        ))

        return final_result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate_result(result: dict[str, Any], max_len: int = 500) -> dict[str, Any]:
    """Create a compact version of a tool result for storage."""
    compact: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, str) and len(v) > max_len:
            compact[k] = v[:max_len] + "…"
        elif isinstance(v, list) and len(v) > 5:
            compact[k] = v[:5]
            compact[f"{k}_total"] = len(v)
        else:
            compact[k] = v
    return compact
