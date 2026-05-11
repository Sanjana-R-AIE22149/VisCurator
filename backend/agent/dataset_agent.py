"""
VisCurator / CVAgent — Dataset Curation Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Conversational ReAct agent. The agent:
1. Asks clarifying questions if the request is vague
2. Searches multiple sources (HuggingFace, Kaggle, Roboflow, PapersWithCode)
3. Presents ranked options and ASKS the user which to pick
4. Estimates quality on the chosen dataset
5. Creates a deterministic preprocessing plan
6. Runs deterministic cleaning/augmentation with live streaming
7. Can be resumed mid-conversation when user replies to a question
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Coroutine
from uuid import UUID

from backend.agent.nim_client import NIMClient
from backend.agent.tools import TOOL_REGISTRY, execute_tool, set_script_emit_callback
from backend.models.schemas import MessageType, PipelineMessage

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 20
EmitFn = Callable[[PipelineMessage], Coroutine[Any, Any, None]]

_SYSTEM_PROMPT = """\
You are CVAgent, an expert AI assistant for computer-vision dataset curation inside VisCurator.

PERSONALITY:
- You are a senior ML engineer helping a colleague. Friendly, expert, concise.
- You never silently make decisions for the user — always present choices.
- You ask ONE clarifying question at a time.

SOURCE AVAILABILITY:
- HuggingFace: always available. PREFER this source for everything.
- PapersWithCode: always available.
- Kaggle: only if KAGGLE credentials are set. If source_status shows "unavailable", skip it.
- Roboflow: only if API key is set. If source_status shows "unavailable", skip it.

WORKFLOW — follow these steps in order:

STEP 1: SEARCH FIRST — ALWAYS
Call search_datasets IMMEDIATELY with the user's raw query.
NEVER ask for clarification before searching.
Even vague queries like "plants", "medical", "cars" should be searched first.
The search system will handle query expansion automatically.
Use sources=["huggingface"] always.

If search returns 0 results:
  - Try search_datasets again with a shorter version of the query (1-2 key words)
  - If still 0, THEN ask one targeted clarifying question

If search returns results but they seem off-topic:
  - Still present them to the user — let the user judge relevance
  - Add a note in your recommendation explaining what you found
  - Do NOT recommend a dataset for a different crop/species as the best match.
    For example, for "apple leaf disease", cassava/tomato/mango/grape datasets
    are fallback options only, not recommendations.

NEVER call ask_clarification as your first action.
NEVER say "I couldn't find datasets" without trying at least 2 searches.

STEP 2: PRESENT OPTIONS
After searching, call present_dataset_options with the top results.
This pauses the pipeline. The user will select a dataset ID.
If the user asked for a specific crop/species and the results are for other
crops/species, say so clearly in the recommendation and ask whether to broaden.

STEP 3: POST-SELECTION ANALYSIS (HEAVY)
Only AFTER the user selects a specific dataset_id:
1. Call estimate_dataset_quality (now that we have one target).
2. Call analyze_dataset_and_plan_processing using the results from quality estimation.
3. Call clean_and_augment_dataset with the final plan.

STEP 4: SUMMARIZE
Give a final report of the curated dataset.

CRITICAL RULES:
- ALWAYS call present_dataset_options before downloading.
- Extract dataset_id EXACTLY from the user's selection message — do not modify or guess it.
- If clean_and_augment_dataset fails with a label column error, retry with label_column="label" instead of "labels".
- If it fails again, retry with label_column="fine_label".
- Never give up after one failure — try the alternatives.
- Keep thoughts to 1-2 sentences max.
"""


def _tools_as_prompt_text() -> str:
    lines = [
        "TOOLS — call a tool by outputting this block (nothing else on those lines):\n",
        "<tool_call>",
        '{"name": "tool_name", "arguments": {"arg1": "value1"}}',
        "</tool_call>\n",
        "AVAILABLE TOOLS:\n",
    ]
    for entry in TOOL_REGISTRY.values():
        params = entry["parameters"].get("properties", {})
        required = entry["parameters"].get("required", [])
        param_desc = ", ".join(
            f'{k}{"*" if k in required else ""} ({v.get("type", "any")}): {v.get("description", "")}'
            for k, v in params.items()
        )
        lines.append(f'[{entry["name"]}] {entry["description"]}')
        lines.append(f'  Params: {param_desc}\n')
    return "\n".join(lines)


def _extract_tool_calls(text: str) -> list[dict]:
    """Extract <tool_call>JSON</tool_call> blocks from LLM output."""
    calls = []
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            if "name" in call and "arguments" in call:
                calls.append(call)
        except json.JSONDecodeError:
            continue
    return calls


def _summarise_search_result(result: dict[str, Any]) -> str:
    """Convert a search result into a compact, LLM-readable summary.
    
    This is the KEY fix — instead of truncating JSON mid-array, we build
    a structured text summary the LLM can actually read and reason about.
    """
    datasets = result.get("datasets", [])
    source_status = result.get("source_status", {})
    total = result.get("total_found", 0)

    lines = [f"Search found {total} datasets."]

    # Source status
    status_parts = []
    for src, status in source_status.items():
        status_parts.append(f"{src}={status}")
    if status_parts:
        lines.append(f"Source status: {', '.join(status_parts)}")

    lines.append("")
    lines.append("TOP RESULTS (use these dataset_ids exactly):")

    hf_results = [d for d in datasets if d.get("source") == "HuggingFace" and not d.get("unavailable")]
    other_results = [d for d in datasets if d.get("source") != "HuggingFace" and not d.get("unavailable")]

    shown = 0
    for ds in (hf_results + other_results)[:8]:
        lines.append(
            f"  - dataset_id: {ds['dataset_id']!r} | source: {ds['source']} | "
            f"name: {ds.get('name','')} | downloads: {ds.get('downloads',0):,} | "
            f"likes: {ds.get('likes',0)} | size: {ds.get('size_estimate','unknown')}"
        )
        desc = (ds.get("description") or "")[:120]
        if desc:
            lines.append(f"    desc: {desc}")
        lines.append(f"    url: {ds.get('url','')}")
        shown += 1

    if shown == 0:
        lines.append("  (no results found — try a broader query)")

    return "\n".join(lines)


def _format_tool_result_for_llm(fn_name: str, tool_result: dict[str, Any]) -> str:
    """Format tool results as readable text for the LLM instead of raw JSON.
    
    Raw JSON gets truncated and confuses the model. Structured text is better.
    """
    status = tool_result.get("status", "unknown")

    if status == "error":
        error = tool_result.get("error", "unknown error")
        tail = tool_result.get("output_tail", [])
        tail_str = "\n".join(tail[-5:]) if tail else ""
        msg = f"Tool '{fn_name}' FAILED with error: {error}"
        if tail_str:
            msg += f"\nLast output lines:\n{tail_str}"
        msg += "\nDo NOT retry with identical arguments. Try a different approach."
        return msg

    if fn_name == "search_datasets":
        return f"Tool 'search_datasets' succeeded.\n{_summarise_search_result(tool_result)}"

    if fn_name == "present_dataset_options":
        return "Tool 'present_dataset_options' succeeded. Pipeline is now PAUSED waiting for user selection."

    if fn_name == "ask_clarification":
        return "Tool 'ask_clarification' succeeded. Pipeline is now PAUSED waiting for user answer."

    if fn_name == "get_dataset_info":
        features = tool_result.get("features", [])
        splits = tool_result.get("splits", {})
        split_info = ", ".join(f"{k}={v.get('num_rows',0):,} rows" for k, v in splits.items())
        return (
            f"Tool 'get_dataset_info' succeeded for {tool_result.get('dataset_id')}.\n"
            f"  Features: {features}\n"
            f"  Splits: {split_info}\n"
            f"  Total rows: {tool_result.get('total_rows', 0):,}\n"
            f"  License: {tool_result.get('license', 'unknown')}"
        )

    if fn_name == "estimate_dataset_quality":
        return (
            f"Tool 'estimate_dataset_quality' succeeded.\n"
            f"  Quality score: {tool_result.get('quality_score', 0)}/100 (grade {tool_result.get('quality_grade','?')})\n"
            f"  Blur ratio: {tool_result.get('blur_ratio', 0):.1%}\n"
            f"  Duplicate %: {tool_result.get('duplicate_percentage', 0):.1f}%\n"
            f"  Class balance: {tool_result.get('class_balance', 'unknown')}\n"
            f"  Images sampled: {tool_result.get('images_sampled', 0)}\n"
            f"  Avg resolution: {tool_result.get('avg_resolution', {})}"
        )

    if fn_name == "analyze_dataset_and_plan_processing":
        return (
            f"Tool 'analyze_dataset_and_plan_processing' succeeded.\n"
            f"  Blur filtering needed: {tool_result.get('needs_blur_filtering')}\n"
            f"  Deduplication needed: {tool_result.get('needs_deduplication')}\n"
            f"  Augmentation needed: {tool_result.get('needs_augmentation')}\n"
            f"  Recommended augmentations: {tool_result.get('recommended_augmentations', [])}\n"
            f"  Reasoning: {tool_result.get('reasoning', '')}"
        )

    if fn_name == "clean_and_augment_dataset":
        exit_code = tool_result.get("exit_code", 0)
        tail = tool_result.get("output_tail", [])
        report = tool_result.get("preprocessing_report") or {}
        after = (report.get("after_stats") or {}).get("images", 0)

        if exit_code != 0 or after == 0:
            tail_str = "\n".join(tail[-8:]) if tail else ""
            msg = f"Tool 'clean_and_augment_dataset' FAILED (exit_code={exit_code}, images_exported={after}).\n"
            if tail_str:
                msg += f"Last output:\n{tail_str}\n"
            if after == 0 and exit_code == 0:
                msg += "All images were filtered out — the label_column name may be wrong. "
                msg += "Retry with a different label_column: try 'label' if you used 'labels', or 'fine_label'."
            else:
                msg += "Retry with a different label_column name or a different dataset."
            return msg

        before = (report.get("before_stats") or {}).get("images", 0)
        class_dist = report.get("class_distribution", {})
        out_dir = report.get("output_dir", "unknown")
        return (
            f"Tool 'clean_and_augment_dataset' SUCCEEDED.\n"
            f"  Images before: {before:,}\n"
            f"  Images after: {after:,}\n"
            f"  Classes: {class_dist}\n"
            f"  Output directory: {out_dir}\n"
            f"  Blur filtered: {(report.get('after_stats') or {}).get('blur_filtered', 0)}\n"
            f"  Duplicates removed: {(report.get('after_stats') or {}).get('duplicates_removed', 0)}"
        )

    if fn_name == "generate_and_run_script":
        exit_code = tool_result.get("exit_code", 0)
        tail = tool_result.get("output_tail", [])
        if exit_code != 0:
            tail_str = "\n".join(tail[-8:]) if tail else ""
            return (
                f"Tool 'generate_and_run_script' FAILED (exit_code={exit_code}).\n"
                f"Last output:\n{tail_str}\n"
                f"Do not retry with identical arguments."
            )
        tail_str = "\n".join(tail[-5:]) if tail else ""
        return (
            f"Tool 'generate_and_run_script' SUCCEEDED.\n"
            f"Output dir: {tool_result.get('output_dir', 'unknown')}\n"
            f"Last lines:\n{tail_str}"
        )

    # Fallback: compact JSON, capped at 3000 chars
    compact = json.dumps(tool_result, default=str)
    if len(compact) > 3000:
        compact = compact[:3000] + "...(truncated)"
    return f"Tool '{fn_name}' result:\n{compact}"


class DatasetAgent:
    """Conversational dataset curation agent."""

    def __init__(self, nim_client: NIMClient, job_id: UUID, emit: EmitFn) -> None:
        self._nim = nim_client
        self._job_id = job_id
        self._emit = emit
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": _tools_as_prompt_text() + "\n\n" + _SYSTEM_PROMPT},
        ]
        self._iteration = 0
        self._tool_results: list[dict[str, Any]] = []
        self._paused = False
        self._pause_reason: str | None = None
        self._last_search_result: dict[str, Any] | None = None
        self._last_dataset_options: list[dict[str, Any]] = []
        self._presented_options = False

    def update_emit_callback(self, emit: EmitFn) -> None:
        """Update the callback used for streaming updates."""
        self._emit = emit

    async def run(
        self,
        query: str,
        source: str = "all",
        target_size: int = 1000,
        user_reply: str | None = None,
    ) -> dict[str, Any]:

        if user_reply:
            self._messages.append({"role": "user", "content": user_reply})
            self._paused = False
            await self._emit(PipelineMessage(
                type=MessageType.LOG,
                message=f"Resuming pipeline — user replied: {user_reply[:80]}",
            ))
        else:
            user_msg = (
                f"Find datasets for: {query}\\n"
                f"Target: {target_size} images.\\n"
                f"Call search_datasets now with query={query!r} and sources=['huggingface']."
            )

            self._messages.append({"role": "user", "content": user_msg})
            await self._emit(PipelineMessage(
                type=MessageType.LOG,
                message=f'Pipeline started — "{query}" | target: {target_size} images',
                data={"query": query, "source": source, "target_size": target_size},
            ))

        async def _script_emit(line: str, stream: str = "stdout") -> None:
            await self._emit(PipelineMessage(
                type=MessageType.SCRIPT_LOG, message=line, data={"stream": stream}
            ))

        set_script_emit_callback(_script_emit)
        final_answer = ""

        while self._iteration < MAX_ITERATIONS:
            self._iteration += 1
            logger.info("[ITERATION %d] Calling NIM for next step...", self._iteration)

            await self._emit(PipelineMessage(
                type=MessageType.LOG,
                message=f"Thinking … (step {self._iteration})",
            ))

            try:
                response = await self._nim.chat(
                    messages=self._messages,
                    temperature=0.1,
                    max_tokens=2048,
                    stream=False,
                )
            except Exception as exc:
                logger.exception("[ERROR] LLM call failed at iteration %d", self._iteration)
                await self._emit(PipelineMessage(
                    type=MessageType.DONE,
                    message=f"NIM API error: {exc}",
                    data={"paused": False, "error": str(exc)},
                ))
                break

            content = response.choices[0].message.content or ""
            self._messages.append({"role": "assistant", "content": content})

            tool_calls = _extract_tool_calls(content)

            # Show the thought (text outside tool_call blocks)
            clean = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
            if clean:
                logger.info("[THOUGHT] Agent is thinking: %s", clean[:100] + "..." if len(clean) > 100 else clean)
                await self._emit(PipelineMessage(
                    type=MessageType.THOUGHT,
                    message=clean,
                    data={"iteration": self._iteration},
                ))

            if not tool_calls:
                if self._last_search_result is not None and not self._presented_options:
                    logger.info("[AUTO] Presenting dataset options after search because model returned no tool call.")
                    datasets = [
                        d for d in self._last_search_result.get("datasets", [])
                        if isinstance(d, dict) and not d.get("unavailable")
                    ][:8]
                    options = [
                        {
                            "source": d.get("source", "HuggingFace"),
                            "dataset_id": d.get("dataset_id", ""),
                            "name": d.get("name") or str(d.get("dataset_id", "")).split("/")[-1],
                            "description": d.get("description", ""),
                            "size_estimate": d.get("size_estimate", "unknown"),
                            "url": d.get("url", ""),
                            "pros": ["Matched the search query"],
                            "cons": ["Review relevance before selecting"],
                        }
                        for d in datasets
                    ]
                    if options:
                        fn_name = "present_dataset_options"
                        fn_args = {
                            "options": options,
                            "recommendation": (
                                "Review the matches below and choose the dataset that best fits your task. "
                                "Prefer a crop-specific dataset if you need one."
                            ),
                            "question": "Which dataset would you like me to download and prepare?",
                        }
                        await self._emit(PipelineMessage(
                            type=MessageType.TOOL_CALL,
                            message=f"â†’ {fn_name}",
                            data={"tool": fn_name, "arguments": fn_args, "iteration": self._iteration},
                        ))
                        tool_result = await execute_tool(fn_name, fn_args)
                        self._presented_options = True
                        self._last_dataset_options = list(tool_result.get("options", []))
                        self._tool_results.append({
                            "tool": fn_name,
                            "arguments": fn_args,
                            "result": _truncate_result(tool_result),
                            "iteration": self._iteration,
                        })
                        await self._emit(PipelineMessage(
                            type=MessageType.TOOL_RESULT,
                            message=f"âœ“ {fn_name}",
                            data=tool_result,
                        ))
                        pause_msg = "â¸ Choose a dataset from the options above."
                        self._paused = True
                        self._pause_reason = tool_result.get("type", "dataset_selection")
                        self._messages.append({"role": "user", "content": _format_tool_result_for_llm(fn_name, tool_result)})
                        await self._emit(PipelineMessage(
                            type=MessageType.DONE,
                            message=pause_msg,
                            data={
                                **tool_result,
                                "paused": True,
                                "pause_reason": self._pause_reason,
                                "summary": pause_msg,
                                "tool_calls_made": self._iteration,
                                "tool_results": self._tool_results,
                            },
                        ))
                        return {
                            "paused": True,
                            "pause_reason": self._pause_reason,
                            "summary": pause_msg,
                            "tool_calls_made": self._iteration,
                            "tool_results": self._tool_results,
                        }

                logger.info("[DONE] No more tool calls. Finalizing answer.")
                final_answer = clean
                break

            for call in tool_calls:
                fn_name = call.get("name", "")
                fn_args = call.get("arguments", {})
                if not isinstance(fn_args, dict):
                    fn_args = {}

                # Auto-inject target_size
                if fn_name in ("generate_and_run_script", "clean_and_augment_dataset"):
                    if "target_size" not in fn_args:
                        fn_args["target_size"] = target_size

                logger.info("[TOOL START] Executing tool: %s with args: %r", fn_name, fn_args)
                await self._emit(PipelineMessage(
                    type=MessageType.TOOL_CALL,
                    message=f"→ {fn_name}",
                    data={"tool": fn_name, "arguments": fn_args, "iteration": self._iteration},
                ))

                tool_result = await execute_tool(fn_name, fn_args)
                logger.info("[TOOL END] Tool %s returned status: %s", fn_name, tool_result.get("status", "unknown"))

                if fn_name == "search_datasets" and tool_result.get("status") == "success":
                    self._last_search_result = tool_result
                elif fn_name == "present_dataset_options":
                    self._presented_options = True
                    self._last_dataset_options = list(tool_result.get("options", []))

                self._tool_results.append({
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": _truncate_result(tool_result),
                    "iteration": self._iteration,
                })

                # Surface errors clearly — include output_tail in the message so agent sees WHY
                result_msg = f"✓ {fn_name}"
                if tool_result.get("status") == "error":
                    error_detail = tool_result.get("error", "unknown error")
                    tail = tool_result.get("output_tail", [])
                    tail_str = "\n".join(tail[-10:]) if tail else ""
                    result_msg = f"✗ {fn_name} FAILED: {error_detail}"
                    if tail_str:
                        result_msg += f"\nLast output:\n{tail_str}"
                elif tool_result.get("exit_code", 0) != 0:
                    tail = tool_result.get("output_tail", [])
                    tail_str = "\n".join(tail[-10:]) if tail else ""
                    result_msg = f"✗ {fn_name} exited with code {tool_result['exit_code']}"
                    if tail_str:
                        result_msg += f"\nLast output:\n{tail_str}"
                # Check for silent zero-image failure
                elif fn_name in ("clean_and_augment_dataset", "generate_and_run_script"):
                    report = tool_result.get("preprocessing_report", {})
                    after = (report.get("after_stats") or {}).get("images", -1)
                    tail = tool_result.get("output_tail", [])
                    if after == 0:
                        result_msg = f"✗ {fn_name} produced 0 images — all images were filtered out. "
                        result_msg += "The blur threshold may be too strict for this dataset's resolution. "
                        result_msg += f"Last output: {' | '.join(tail[-5:])}"

                await self._emit(PipelineMessage(
                    type=MessageType.TOOL_RESULT,
                    message=result_msg,
                    data=tool_result,
                ))

                # Pause if waiting for user
                if tool_result.get("status") == "waiting_for_user":
                    self._paused = True
                    self._pause_reason = tool_result.get("type", "unknown")
                    pause_type = tool_result.get("type", "")
                    pause_msg = (
                        "⏸ Choose a dataset from the options above."
                        if pause_type == "dataset_selection"
                        else f"⏸ {tool_result.get('question', 'Please reply.')}"
                    )
                    llm_msg = _format_tool_result_for_llm(fn_name, tool_result)
                    self._messages.append({"role": "user", "content": llm_msg})
                    await self._emit(PipelineMessage(
                        type=MessageType.DONE,
                        message=pause_msg,
                        data={
                            **tool_result,
                            "paused": True,
                            "pause_reason": self._pause_reason,
                            "summary": pause_msg,
                            "tool_calls_made": self._iteration,
                            "tool_results": self._tool_results,
                        },
                    ))
                    return {
                        "paused": True,
                        "pause_reason": self._pause_reason,
                        "summary": pause_msg,
                        "tool_calls_made": self._iteration,
                        "tool_results": self._tool_results,
                    }

                # Format result for LLM using the structured formatter
                # (this replaces raw JSON truncation — _format_tool_result_for_llm
                #  converts each tool result into compact readable text the LLM can parse)
                llm_msg = _format_tool_result_for_llm(fn_name, tool_result)

                if (
                    fn_name == "estimate_dataset_quality"
                    and tool_result.get("status") == "error"
                    and "No configs found" in str(tool_result.get("error", ""))
                ):
                    bad_dataset_id = str(fn_args.get("dataset_id", ""))
                    retry_options = [
                        option for option in self._last_dataset_options
                        if option.get("dataset_id") != bad_dataset_id
                    ]
                    pause_msg = (
                        "This dataset is not compatible with the current HuggingFace loader. "
                        "Please choose a different dataset from the options."
                    )
                    self._paused = True
                    self._pause_reason = "dataset_selection"
                    await self._emit(PipelineMessage(
                        type=MessageType.DONE,
                        message=pause_msg,
                        data={
                            "paused": True,
                            "type": "dataset_selection",
                            "question": "Please select a different dataset. This one cannot be loaded.",
                            "options": retry_options,
                            "pause_reason": self._pause_reason,
                            "summary": pause_msg,
                            "tool_calls_made": self._iteration,
                            "tool_results": self._tool_results,
                        },
                    ))
                    return {
                        "paused": True,
                        "pause_reason": self._pause_reason,
                        "summary": pause_msg,
                        "tool_calls_made": self._iteration,
                        "tool_results": self._tool_results,
                    }
                
                # For tools that produce complex data needed by downstream tools,
                # also append compact JSON so LLM can reference exact values
                if fn_name in ("estimate_dataset_quality", "get_dataset_info", "analyze_dataset_and_plan_processing"):
                    compact_json = json.dumps(tool_result, default=str)
                    if len(compact_json) <= 2000:  # Only include if reasonably sized
                        llm_msg += f"\n\n[RAW DATA for tool chaining]:\n{compact_json}"
                
                self._messages.append({"role": "user", "content": llm_msg})

                # Prompt the LLM to continue
                self._messages.append({
                    "role": "user",
                    "content": (
                        "Good. Continue with the next step of the workflow. "
                        "If you need to call another tool, output a <tool_call> block now. "
                        "If you are completely done with ALL steps (curation finished), "
                        "write your final summary with no tool calls."
                    )
                })

        if not final_answer and self._iteration >= MAX_ITERATIONS:
            try:
                self._messages.append({
                    "role": "user",
                    "content": "Max steps reached. Give your best final summary now. No tool calls.",
                })
                resp2 = await self._nim.chat(
                    messages=self._messages, temperature=0.1, max_tokens=512, stream=False
                )
                final_answer = resp2.choices[0].message.content or "Pipeline complete."
            except Exception:
                final_answer = "Pipeline complete."

        result = {
            "paused": False,
            "summary": final_answer,
            "tool_calls_made": self._iteration,
            "tool_results": self._tool_results,
        }
        await self._emit(PipelineMessage(
            type=MessageType.DONE,
            message=final_answer,
            data=result,
        ))
        return result


def _truncate_result(result: dict[str, Any], max_len: int = 400) -> dict[str, Any]:
    """Compact a result dict for storage (not for LLM — use _format_tool_result_for_llm instead)."""
    compact: dict[str, Any] = {}
    for k, v in result.items():
        if k in ("blur_scatter", "output_tail"):
            compact[k] = v[:5] if isinstance(v, list) else v
        elif isinstance(v, str) and len(v) > max_len:
            compact[k] = v[:max_len] + "…"
        elif isinstance(v, list) and len(v) > 5:
            compact[k] = v[:5]
        else:
            compact[k] = v
    return compact
