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
6. Can be resumed mid-conversation when user replies to a question
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine
from uuid import UUID

from backend.agent.nim_client import NIMClient
from backend.agent.tools import TOOL_REGISTRY, execute_tool, set_script_emit_callback
from backend.models.schemas import MessageType, PipelineMessage

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 15
EmitFn = Callable[[PipelineMessage], Coroutine[Any, Any, None]]

_SYSTEM_PROMPT = """\
You are **CVAgent**, an expert AI assistant for computer-vision dataset curation inside VisCurator.

YOUR PERSONALITY:
- Friendly, expert, concise. Think of yourself as a senior ML engineer helping a colleague.
- You ask ONE clarifying question at a time, never a wall of questions.
- You never silently make decisions for the user — you always present choices.

SOURCE AVAILABILITY:
- HuggingFace: always available, no key required. Prefer this source.
- PapersWithCode: always available, no key required.
- Kaggle: requires KAGGLE_USERNAME + KAGGLE_KEY in .env. If the search result contains
  an entry with dataset_id="unavailable" from Kaggle, that source is not configured —
  do NOT suggest Kaggle datasets or ask the user to set up Kaggle unless they ask.
- Roboflow: requires ROBOFLOW_API_KEY in .env. Same rule — if unavailable, skip it silently.
When a source is unavailable, mention it briefly once (e.g. "Kaggle is not configured") and
focus on the available sources. Never waste iterations retrying an unavailable source.

YOUR WORKFLOW (follow this order strictly):

STEP 1 — CLARIFY (if needed)
If the user's request is vague (e.g. "disease detection" without specifying plant/human/animal, 
or "object detection" without specifying what objects), call `ask_clarification` with specific options.
If the request is clear enough, skip to Step 2.

STEP 2 — SEARCH
Call `search_datasets` with a refined query. Search ALL sources by default (huggingface, kaggle, roboflow, paperswithcode).
Use the full query including any details the user confirmed in Step 1.
Check `source_status` in the result — skip any source marked "unavailable" in subsequent steps.

STEP 3 — INSPECT TOP RESULTS  
For the 2-3 best HuggingFace results, call `get_dataset_info` to get exact split sizes and features.
For Kaggle/Roboflow results, you already have enough from the search.

STEP 4 — QUALITY CHECK (optional but recommended)
Call `estimate_dataset_quality` on the top 1-2 HuggingFace candidates to get blur ratio and quality score.

STEP 5 — PRESENT OPTIONS AND ASK
Call `present_dataset_options` with 3-5 curated options (only from available sources).
Include pros/cons for each. This PAUSES the pipeline — the user will click to choose.
NEVER skip this step and silently pick a dataset yourself.

STEP 6 — DOWNLOAD (after user confirms)
Once the user has selected a dataset (their message will say which one they picked),
call `get_dataset_info` and `estimate_dataset_quality` for the selected dataset if needed,
then call `analyze_dataset_and_plan_processing` using blur score, duplicate percentage,
class balance, dataset size, and resolution stats.
After that, call `clean_and_augment_dataset` with the correct dataset_id, source,
label_column, image_column, and processing_plan.
Infer label_column and image_column from get_dataset_info results (look at features).

STEP 7 — SUMMARIZE
After processing completes, give a clear summary: dataset chosen, preprocessing decisions,
before/after image counts, class distribution, output location, next steps.

RULES:
- ALWAYS call present_dataset_options before downloading. No exceptions.
- Never invent dataset IDs — only use IDs from search results.
- If a tool returns an error, try an alternative approach or dataset.
- When the user replies to a question (e.g. "use option 2" or "I want the Kaggle one"), 
  continue from wherever you paused — do NOT restart from Step 1.
- Keep your thought messages concise (1-2 sentences). Details go in tool calls.
"""


def _nim_tools() -> list[dict[str, Any]]:
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


class DatasetAgent:
    """Conversational dataset curation agent.
    
    Supports mid-session resumption: call run() again with user_reply
    to continue after a pause (ask_clarification or present_dataset_options).
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
        self._paused = False
        self._pause_reason: str | None = None

    # ── Public ────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        source: str = "all",
        target_size: int = 1000,
        user_reply: str | None = None,
    ) -> dict[str, Any]:
        """Run or resume the pipeline.
        
        Parameters
        ----------
        query : str
            Initial user request (on first call) or the full current message.
        user_reply : str | None
            If resuming after a pause (user answered a question), pass their reply here.
        """

        if user_reply:
            # Resume: inject user reply and continue
            self._messages.append({"role": "user", "content": user_reply})
            self._paused = False
            await self._emit(PipelineMessage(
                type=MessageType.LOG,
                message=f"Resuming pipeline — user replied: {user_reply[:80]}",
            ))
        else:
            # Fresh start
            user_msg = (
                f"I need a curated computer-vision dataset.\n"
                f"Task/topic: {query}\n"
                f"Target images: {target_size}\n\n"
                f"Please help me find, evaluate and download the best dataset for this."
            )
            self._messages.append({"role": "user", "content": user_msg})
            await self._emit(PipelineMessage(
                type=MessageType.LOG,
                message=f'Pipeline started — "{query}" | target: {target_size} images',
                data={"query": query, "source": source, "target_size": target_size},
            ))

        async def _script_line_emit(line: str, stream: str = "stdout") -> None:
            await self._emit(PipelineMessage(type=MessageType.SCRIPT_LOG, message=line, data={"stream": stream}))

        set_script_emit_callback(_script_line_emit)
        tools = _nim_tools()
        final_answer = ""

        while self._iteration < MAX_ITERATIONS:
            self._iteration += 1

            await self._emit(PipelineMessage(
                type=MessageType.LOG,
                message=f"Thinking … (step {self._iteration})",
            ))

            try:
                response = await self._nim.chat(
                    messages=self._messages,
                    temperature=0.15,
                    max_tokens=2048,
                    stream=False,
                    tools=tools,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.exception("LLM call failed at iteration %d", self._iteration)
                await self._emit(PipelineMessage(type=MessageType.ERROR, message=f"LLM error: {exc}"))
                break

            choice = response.choices[0]
            msg = choice.message

            assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            self._messages.append(assistant_entry)

            if msg.content and msg.content.strip():
                await self._emit(PipelineMessage(
                    type=MessageType.THOUGHT,
                    message=msg.content.strip(),
                    data={"iteration": self._iteration},
                ))

            if not msg.tool_calls:
                final_answer = msg.content or ""
                break

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # Inject target_size into download script
                if fn_name == "generate_and_run_script" and "target_size" not in fn_args:
                    fn_args["target_size"] = target_size

                await self._emit(PipelineMessage(
                    type=MessageType.TOOL_CALL,
                    message=f"→ {fn_name}",
                    data={"tool": fn_name, "arguments": fn_args, "iteration": self._iteration},
                ))

                tool_result = await execute_tool(fn_name, fn_args)

                self._tool_results.append({
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": _truncate_result(tool_result),
                    "iteration": self._iteration,
                })

                await self._emit(PipelineMessage(
                    type=MessageType.TOOL_RESULT,
                    message=f"✓ {fn_name}",
                    data=tool_result,
                ))

                # Check if the tool is requesting user input — PAUSE here
                if tool_result.get("status") == "waiting_for_user":
                    self._paused = True
                    self._pause_reason = tool_result.get("type", "unknown")

                    pause_type = tool_result.get("type", "")
                    pause_msg = (
                        "⏸ Waiting for your input — check the options above and reply to continue."
                        if pause_type == "dataset_selection"
                        else f"⏸ {tool_result.get('question', 'Please reply to continue.')}"
                    )

                    # Feed the result back so the agent knows what happened
                    result_text = json.dumps(tool_result, default=str)
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

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

                result_text = json.dumps(tool_result, default=str)
                if len(result_text) > 6000:
                    result_text = result_text[:6000] + "\n...(truncated)"

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

        # Max iterations fallback
        if not final_answer and self._iteration >= MAX_ITERATIONS:
            try:
                self._messages.append({
                    "role": "user",
                    "content": "Provide your best final summary now, no more tool calls.",
                })
                resp2 = await self._nim.chat(messages=self._messages, temperature=0.1, max_tokens=1024, stream=False)
                final_answer = resp2.choices[0].message.content or ""
            except Exception:
                final_answer = "Agent reached max iterations."

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
    compact: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, str) and len(v) > max_len:
            compact[k] = v[:max_len] + "…"
        elif isinstance(v, list) and len(v) > 5:
            compact[k] = v[:5]
        else:
            compact[k] = v
    return compact
