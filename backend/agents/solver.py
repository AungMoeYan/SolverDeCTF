"""Efficient per-model CTF solver.

Architecture:

    Bedrock model
        ↓
    bounded Pydantic-AI tool loop
        ↓
    Docker/Kali sandbox
        ↓
    tool result
        ↓
    model
        ↓
    flag / next action

The solver deliberately keeps model runs bounded and aggressively
terminates once a flag has been confirmed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset
from pydantic_ai.usage import UsageLimits

from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.deps import SolverDeps
from backend.investigation import (
    ActionTracker,
    InvestigationPhase,
    STALL_MESSAGE,
    action_signature,
    apply_tool_result,
    compact_summary,
    extract_flag_candidate,
    mark_confirmed_flag,
    orchestration_system_prompt,
    should_inspect_distfiles,
    state_from_challenge,
    summarize_tool_result,
)
from backend.loop_detect import (
    LOOP_WARNING_MESSAGE,
    LoopDetector,
)
from backend.models import (
    model_id_from_spec,
    provider_from_spec,
    resolve_model,
    resolve_model_settings,
    supports_vision,
)
from backend.output_types import FlagFound
from backend.prompts import (
    ChallengeMeta,
    build_prompt,
    list_distfiles,
)
from backend.sandbox import DockerSandbox
from backend.solver_base import (
    CANCELLED,
    ERROR,
    FLAG_FOUND,
    GAVE_UP,
    SolverResult,
)
from backend.tools.flag import submit_flag
from backend.tools.sandbox import (
    bash,
    check_findings,
    inspect_distfiles,
    list_files,
    notify_coordinator,
    read_file,
    web_fetch,
    webhook_create,
    webhook_get_requests,
    write_file,
)
from backend.tools.vision import view_image
from backend.tracing import SolverTracer


logger = logging.getLogger(__name__)


# ============================================================================
# FLAG DETECTION
# ============================================================================

def _extract_flag_candidate(
    text: str,
) -> str | None:
    """Extract a likely CTF flag from tool output."""

    return extract_flag_candidate(text)


# ============================================================================
# EFFICIENCY CONTROLS
# ============================================================================

MAX_REQUESTS_PER_RUN = 30

DEFAULT_TOOL_TIMEOUT = 60

MAX_SIBLING_INSIGHTS = 3000

MAX_PROGRESS_TEXT = 1000


# ============================================================================
# HARD STOP
# ============================================================================

class FlagConfirmedSignal(Exception):
    """Internal signal used to immediately terminate an LLM run."""

    def __init__(
        self,
        flag: str,
    ) -> None:
        self.flag = flag

        super().__init__(
            f"Flag confirmed: {flag}"
        )


# ============================================================================
# TOOLSET WRAPPER
# ============================================================================

@dataclass
class TracingToolset(
    WrapperToolset[SolverDeps]
):
    """Wrap tools with tracing, loop detection and flag handling."""

    tracer: SolverTracer = field(
        repr=False
    )

    loop_detector: LoopDetector = field(
        repr=False
    )

    action_tracker: ActionTracker = field(
        repr=False
    )

    step_counter: list[int] = field(
        repr=False
    )

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[SolverDeps],
        tool: ToolsetTool[SolverDeps],
    ) -> Any:
        """Trace and execute one tool call."""

        # ------------------------------------------------------------------
        # STEP COUNT
        # ------------------------------------------------------------------

        self.step_counter[0] += 1

        step = self.step_counter[0]

        self.tracer.tool_call(
            name,
            tool_args,
            step,
        )

        logger.info(
            "[%s] Step %d → %s %s",
            ctx.deps.challenge_name,
            step,
            name,
            _format_tool_args(tool_args),
        )

        state = ctx.deps.investigation

        # ------------------------------------------------------------------
        # IRRELEVANT DISTFILES (web / remote-first challenges)
        # ------------------------------------------------------------------

        if (
            name == "inspect_distfiles"
            and state is not None
            and not should_inspect_distfiles(state)
        ):
            message = (
                "SKIPPED inspect_distfiles. This challenge has a remote "
                "attack surface. Investigate the target URL first. Local "
                "distfiles are irrelevant unless new evidence says otherwise."
            )
            logger.info(
                "[%s] Step %d blocked inspect_distfiles (web/remote priority)",
                ctx.deps.challenge_name,
                step,
            )
            return message

        # ------------------------------------------------------------------
        # DUPLICATE ACTION PREVENTION
        # ------------------------------------------------------------------

        signature = action_signature(name, tool_args)

        if (
            state is not None
            and name != "submit_flag"
            and self.action_tracker.is_duplicate(state, signature)
        ):
            state.consecutive_no_progress += 1
            message = (
                "DUPLICATE ACTION BLOCKED.\n"
                f"Already performed: {signature}\n"
                "Do not repeat this unless privileges or the target changed.\n"
                f"Current phase: {state.phase.value}\n"
                f"Current objective: {state.current_objective}\n"
                f"Rejected hypotheses: {state.rejected_statements() or '(none)'}\n"
            )
            if state.consecutive_no_progress >= 4:
                message += "\n" + STALL_MESSAGE
                state.set_phase(InvestigationPhase.STUCK)
            logger.info(
                "[%s] Step %d duplicate blocked: %s",
                ctx.deps.challenge_name,
                step,
                signature,
            )
            return message

        # ------------------------------------------------------------------
        # LOOP DETECTION
        # ------------------------------------------------------------------

        loop_status = self.loop_detector.check(
            name,
            tool_args,
        )

        if loop_status == "break":
            logger.warning(
                "[%s] Loop break on %s at step %d",
                ctx.deps.challenge_name,
                name,
                step,
            )

            self.tracer.event(
                "loop_break",
                tool=name,
                step=step,
            )

            return LOOP_WARNING_MESSAGE

        # ------------------------------------------------------------------
        # EXECUTE TOOL
        # ------------------------------------------------------------------

        try:
            result = await self.wrapped.call_tool(
                name,
                tool_args,
                ctx,
                tool,
            )

        except FlagConfirmedSignal:
            flag = (
                ctx.deps.confirmed_flag
                or ""
            )

            self.tracer.event(
                "flag_hard_stop",
                tool=name,
                step=step,
                flag=flag,
            )

            logger.info(
                "[%s] HARD STOP — flag confirmed at step %d: %s",
                ctx.deps.challenge_name,
                step,
                flag,
            )

            raise

        except Exception as exc:
            self.tracer.tool_result(
                name,
                f"ERROR: {exc}",
                step,
            )

            logger.error(
                "[%s] Step %d ← %s ERROR: %s",
                ctx.deps.challenge_name,
                step,
                name,
                exc,
            )

            raise

        # ------------------------------------------------------------------
        # NORMALIZE RESULT
        # ------------------------------------------------------------------

        result_str = (
            str(result)
            if result is not None
            else ""
        )

        if state is not None:
            state.last_full_results[signature] = result_str[:20000]
            apply_tool_result(
                state,
                name,
                tool_args,
                result_str,
            )
            self.action_tracker.record(
                state,
                name,
                signature,
            )
            result_str = summarize_tool_result(
                name,
                tool_args,
                result_str,
            )
            if state.phase == InvestigationPhase.STUCK:
                result_str += "\n\n" + STALL_MESSAGE
            result_str += (
                "\n\n[state] "
                f"phase={state.phase.value} "
                f"objective={state.current_objective} "
                f"user={state.privileges.get('execution_user', 'unknown')}"
            )
            result = result_str

        self.tracer.tool_result(
            name,
            result_str[:12000],
            step,
        )

        # ------------------------------------------------------------------
        # TERMINAL LOGGING
        # ------------------------------------------------------------------

        logger.info(
            "[%s] Step %d ← %s: %s",
            ctx.deps.challenge_name,
            step,
            name,
            _compact_text(result_str),
        )

        # ------------------------------------------------------------------
        # NORMAL SUBMIT_FLAG HARD STOP
        # ------------------------------------------------------------------

        if (
            name == "submit_flag"
            and ctx.deps.confirmed_flag
        ):
            flag = ctx.deps.confirmed_flag

            self.tracer.event(
                "flag_confirmed",
                tool=name,
                step=step,
                flag=flag,
            )

            self.tracer.event(
                "flag_hard_stop",
                tool=name,
                step=step,
                flag=flag,
            )

            logger.info(
                "[%s] HARD STOP — submit_flag confirmed flag: %s",
                ctx.deps.challenge_name,
                flag,
            )

            raise FlagConfirmedSignal(
                flag
            )

        # ------------------------------------------------------------------
        # FLAG DETECTION
        # ------------------------------------------------------------------

        candidate = _extract_flag_candidate(
            result_str
        )

        if (
            candidate
            and name != "submit_flag"
            and not ctx.deps.confirmed_flag
        ):
            self.tracer.event(
                "flag_candidate_detected",
                tool=name,
                step=step,
                candidate=candidate,
            )

            logger.info(
                "[%s] FLAG CANDIDATE detected at step %d: %s",
                ctx.deps.challenge_name,
                step,
                candidate,
            )

            if ctx.deps.no_submit:
                ctx.deps.confirmed_flag = candidate
                if state is not None:
                    mark_confirmed_flag(state, candidate)
                logger.info(
                    "[%s] HARD STOP — no-submit candidate accepted: %s",
                    ctx.deps.challenge_name,
                    candidate,
                )
                raise FlagConfirmedSignal(candidate)

            result_str += (
                "\n\n"
                "IMPORTANT: A likely CTF flag was detected in this "
                "tool output:\n"
                f"{candidate}\n\n"
                "Immediately call submit_flag with this exact candidate "
                "to verify it. Do NOT perform another reconnaissance "
                "action before verification. Do not continue after a "
                "confirmed flag."
            )

            result = result_str

        # ------------------------------------------------------------------
        # LOOP WARNING
        # ------------------------------------------------------------------

        if loop_status == "warn":
            warning = (
                "\n\n"
                f"{LOOP_WARNING_MESSAGE}"
            )

            if isinstance(
                result,
                str,
            ):
                result = (
                    result
                    + warning
                )

        return result


# ============================================================================
# LOGGING HELPERS
# ============================================================================

def _compact_text(
    text: str,
    limit: int = 500,
) -> str:
    """Make tool output suitable for terminal progress logs."""

    if not text:
        return "(no output)"

    cleaned = " ".join(
        text.replace(
            "\x00",
            "",
        ).split()
    )

    if len(cleaned) <= limit:
        return cleaned

    return (
        cleaned[:limit]
        + "..."
    )


def _format_tool_args(
    args: dict[str, Any],
) -> str:
    """Create a compact representation of tool arguments."""

    if not args:
        return ""

    parts: list[str] = []

    for key, value in args.items():
        value_text = str(value)

        value_text = value_text.replace(
            "\n",
            " ",
        )

        if len(value_text) > 250:
            value_text = (
                value_text[:250]
                + "..."
            )

        parts.append(
            f"{key}={value_text!r}"
        )

    return (
        "("
        + ", ".join(parts)
        + ")"
    )


# ============================================================================
# TOOLSET
# ============================================================================

def _build_toolset(
    deps: SolverDeps,
) -> FunctionToolset[SolverDeps]:
    """Build the available CTF tools.

    Tool order is a priority hint. Remote/web challenges put HTTP
    tools first; file-based challenges keep distfile inspection first.
    """

    remote_first = (
        deps.investigation is not None
        and not should_inspect_distfiles(deps.investigation)
    )

    rest = [
        bash,
        read_file,
        write_file,
        list_files,
        submit_flag,
        webhook_create,
        webhook_get_requests,
        check_findings,
        notify_coordinator,
    ]

    if remote_first:
        tools = [web_fetch, *rest, inspect_distfiles]
    else:
        tools = [inspect_distfiles, *rest, web_fetch]

    if deps.use_vision:
        tools.append(
            view_image
        )

    return FunctionToolset(
        tools=tools,
        max_retries=2,
    )


# ============================================================================
# SOLVER
# ============================================================================

class Solver:
    """One model + one sandbox + one challenge."""

    def __init__(
        self,
        model_spec: str,
        challenge_dir: str,
        meta: ChallengeMeta,
        ctfd: CTFdClient,
        cost_tracker: CostTracker,
        settings: object,
        cancel_event: asyncio.Event | None = None,
        sandbox: DockerSandbox | None = None,
        owns_sandbox: bool | None = None,
    ) -> None:

        self.model_spec = model_spec

        self.model_id = model_id_from_spec(
            model_spec
        )

        self.challenge_dir = challenge_dir

        self.meta = meta

        self.ctfd = ctfd

        self.cost_tracker = cost_tracker

        self.settings = settings

        self.cancel_event = (
            cancel_event
            or asyncio.Event()
        )

        self._owns_sandbox = (
            owns_sandbox
            if owns_sandbox is not None
            else sandbox is None
        )

        self.sandbox = (
            sandbox
            or DockerSandbox(
                image=getattr(
                    settings,
                    "sandbox_image",
                    "ctf-sandbox",
                ),
                challenge_dir=challenge_dir,
                memory_limit=getattr(
                    settings,
                    "container_memory_limit",
                    "4g",
                ),
            )
        )

        self.use_vision = supports_vision(
            model_spec
        )

        instruction = getattr(
            settings,
            "solver_instruction",
            "",
        )

        self.deps = SolverDeps(
            sandbox=self.sandbox,
            ctfd=ctfd,
            challenge_dir=challenge_dir,
            challenge_name=meta.name,
            workspace_dir="",
            use_vision=self.use_vision,
            cost_tracker=cost_tracker,
            instruction=instruction.strip(),
        )

        self.loop_detector = LoopDetector()

        self.action_tracker = ActionTracker()

        self.tracer = SolverTracer(
            meta.name,
            self.model_id,
        )

        self.agent_name = (
            f"{meta.name}/{self.model_id}"
        )

        self._agent: (
            Agent[SolverDeps, FlagFound]
            | None
        ) = None

        self._messages: list[Any] = []

        self._step_count = [0]

        self._flag: str | None = None

        self._confirmed = False

        self._findings = ""

        self._run_number = 0

    # ========================================================================
    # START
    # ========================================================================

    async def start(self) -> None:
        """Start sandbox and construct Pydantic-AI agent."""

        if not self.sandbox._container:
            await self.sandbox.start()

        self.deps.workspace_dir = (
            self.sandbox.workspace_dir
        )

        # ------------------------------------------------------------------
        # Container architecture
        # ------------------------------------------------------------------

        arch_result = await self.sandbox.exec(
            "uname -m",
            timeout_s=10,
        )

        container_arch = (
            arch_result.stdout.strip()
            or "unknown"
        )

        # ------------------------------------------------------------------
        # Distfile metadata
        # ------------------------------------------------------------------

        distfile_names = list_distfiles(
            self.challenge_dir
        )

        hint = str(
            getattr(self.settings, "solver_hint", "") or ""
        ).strip()
        if not hint:
            hint = "\n".join(
                str(item.get("content", ""))
                for item in self.meta.hints
                if item.get("content")
            )
        evidence_paths = list(
            getattr(self.settings, "solver_evidence", []) or []
        )
        description = (
            str(getattr(self.settings, "solver_description", "") or "").strip()
            or self.meta.description
        )
        self.deps.investigation = state_from_challenge(
            description=description,
            hint=hint,
            evidence_paths=evidence_paths,
            metadata_category=self.meta.category,
            challenge_name=self.meta.name,
            connection_info=self.meta.connection_info,
        )
        prompt_meta = self.meta
        if (
            not prompt_meta.connection_info.strip()
            and self.deps.investigation.target_urls
        ):
            prompt_meta = replace(
                self.meta,
                connection_info=self.deps.investigation.target_urls[0],
                category=self.deps.investigation.category.value,
            )
        elif self.meta.name == "interactive":
            prompt_meta = replace(
                self.meta,
                category=self.deps.investigation.category.value,
            )

        # ------------------------------------------------------------------
        # System prompt
        # ------------------------------------------------------------------

        system_prompt = (
            build_prompt(
                prompt_meta,
                distfile_names,
                container_arch=container_arch,
            )
            + "\n\n"
            + orchestration_system_prompt(self.deps.investigation)
        )

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------

        model = resolve_model(
            self.model_spec,
            self.settings,
        )

        model_settings = resolve_model_settings(
            self.model_spec
        )

        # ------------------------------------------------------------------
        # Tools
        # ------------------------------------------------------------------

        raw_toolset = _build_toolset(
            self.deps
        )

        toolset = TracingToolset(
            wrapped=raw_toolset,
            tracer=self.tracer,
            loop_detector=self.loop_detector,
            action_tracker=self.action_tracker,
            step_counter=self._step_count,
        )

        # ------------------------------------------------------------------
        # Agent
        # ------------------------------------------------------------------

        self._agent = Agent(
            model,
            deps_type=SolverDeps,
            system_prompt=system_prompt,
            model_settings=model_settings,
            toolsets=[toolset],
            output_type=FlagFound,
        )

        self.tracer.event(
            "start",
            challenge=self.meta.name,
            model=self.model_id,
            instruction=self.deps.instruction[:1000],
        )

        logger.info(
            "[%s] Solver started",
            self.agent_name,
        )

        if self.deps.instruction:
            logger.info(
                "[%s] Operator instruction: %s",
                self.agent_name,
                _compact_text(
                    self.deps.instruction,
                    1000,
                ),
            )

    # ========================================================================
    # RUN
    # ========================================================================

    async def run_until_done_or_gave_up(
        self,
    ) -> SolverResult:
        """Execute one bounded solver attempt."""

        if not self._agent:
            await self.start()

        assert self._agent is not None

        # ------------------------------------------------------------------
        # Cancellation
        # ------------------------------------------------------------------

        if self.cancel_event.is_set():
            return self._result(
                CANCELLED
            )

        # ------------------------------------------------------------------
        # HARD STOP SAFETY CHECK
        # ------------------------------------------------------------------

        if self.deps.confirmed_flag:
            self._flag = (
                self.deps.confirmed_flag
            )

            self._confirmed = True

            self._findings = (
                f"Flag confirmed: {self._flag}"
            )

            return self._result(
                FLAG_FOUND,
                run_steps=0,
            )

        self._run_number += 1

        t0 = time.monotonic()

        run_steps_before = (
            self._step_count[0]
        )

        # ------------------------------------------------------------------
        # PROMPT
        # ------------------------------------------------------------------

        state = self.deps.investigation

        briefing = (
            compact_summary(state)
            if state is not None
            else ""
        )

        if self._messages:
            prompt = (
                "Continue from the investigation summary. "
                "Do not restart reconnaissance.\n\n"
                f"{briefing}\n\n"
                "Take the single highest-value unexplored action.\n"
                "If a flag-like value appears, verify it immediately.\n"
                "When the flag is confirmed, stop."
            )
        elif state is not None and not should_inspect_distfiles(state):
            prompt = (
                "Solve this CTF challenge.\n\n"
                f"{briefing}\n\n"
                "Start with the remote target. Use tools immediately.\n"
                "Do not inspect /challenge/distfiles unless evidence "
                "later makes local files relevant.\n"
                "Do not guess the flag. When a flag is found, stop."
            )
        else:
            prompt = (
                "Solve this CTF challenge.\n\n"
                f"{briefing}\n\n"
                "Start immediately with the highest-value action.\n"
                "For file-based challenges, begin with inspect_distfiles.\n"
                "Use tools rather than explaining what you might do.\n"
                "Do not guess the flag.\n"
                "When a flag is found, stop."
            )

        # ------------------------------------------------------------------
        # OPERATOR INSTRUCTION
        # ------------------------------------------------------------------

        if self.deps.instruction:
            prompt += (
                "\n\n"
                "OPERATOR INSTRUCTION:\n"
                f"{self.deps.instruction}\n\n"
                "Use this instruction to guide your investigation. "
                "You may use additional techniques when useful. "
                "Always verify candidate flags before reporting them."
            )

        # ------------------------------------------------------------------
        # PROGRESS
        # ------------------------------------------------------------------

        logger.info(
            "[%s] Run %d starting",
            self.agent_name,
            self._run_number,
        )

        # ------------------------------------------------------------------
        # MODEL RUN
        # ------------------------------------------------------------------

        try:
            result = await self._agent.run(
                prompt,
                deps=self.deps,
                message_history=(
                    self._messages
                    if self._messages
                    else None
                ),
                usage_limits=UsageLimits(
                    request_limit=MAX_REQUESTS_PER_RUN,
                ),
            )

        except FlagConfirmedSignal as signal:

            self._flag = signal.flag.strip()

            self._confirmed = True

            self._findings = (
                f"Flag confirmed via submit_flag: "
                f"{self._flag}"
            )

            if self.deps.investigation is not None:
                mark_confirmed_flag(
                    self.deps.investigation,
                    self._flag,
                )

            logger.info(
                "[%s] FLAG CONFIRMED — stopping model immediately: %s",
                self.agent_name,
                self._flag,
            )

            self.tracer.event(
                "hard_stop_success",
                flag=self._flag,
                run_number=self._run_number,
            )

            run_steps = (
                self._step_count[0]
                - run_steps_before
            )

            return self._result(
                FLAG_FOUND,
                run_steps=run_steps,
            )

        except asyncio.CancelledError:

            return self._result(
                CANCELLED,
                run_steps=(
                    self._step_count[0]
                    - run_steps_before
                ),
            )

        except UsageLimitExceeded as exc:

            duration = (
                time.monotonic()
                - t0
            )

            summary = (
                compact_summary(state)
                if state is not None
                else str(exc)
            )

            self._findings = summary

            logger.warning(
                "[%s] Request limit reached — compacting with "
                "preserved investigation state (%.1fs)",
                self.agent_name,
                duration,
            )

            self.tracer.event(
                "usage_limit",
                error=str(exc),
                duration_seconds=round(
                    duration,
                    3,
                ),
                run_number=self._run_number,
            )

            return self._result(
                GAVE_UP,
                run_steps=(
                    self._step_count[0]
                    - run_steps_before
                ),
            )

        except Exception as exc:

            duration = (
                time.monotonic()
                - t0
            )

            logger.error(
                "[%s] Error: %s",
                self.agent_name,
                exc,
                exc_info=True,
            )

            self._findings = (
                f"Error during solver run: {exc}"
            )

            self.tracer.event(
                "error",
                error=str(exc),
                duration_seconds=round(
                    duration,
                    3,
                ),
                run_number=self._run_number,
            )

            return self._result(
                ERROR,
                run_steps=(
                    self._step_count[0]
                    - run_steps_before
                ),
            )

        # ------------------------------------------------------------------
        # USAGE
        # ------------------------------------------------------------------

        duration = (
            time.monotonic()
            - t0
        )

        usage = result.usage

        try:
            self.cost_tracker.record(
                self.agent_name,
                usage,
                self.model_id,
                provider_spec=provider_from_spec(
                    self.model_spec
                ),
                duration_seconds=duration,
            )

        except Exception as exc:
            logger.warning(
                "Could not record cost for %s: %s",
                self.model_id,
                exc,
            )

        agent_usage = (
            self.cost_tracker.by_agent.get(
                self.agent_name
            )
        )

        self.tracer.usage(
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            (
                agent_usage.cost_usd
                if agent_usage
                else 0.0
            ),
        )

        # ------------------------------------------------------------------
        # SAVE HISTORY
        # ------------------------------------------------------------------

        self._messages = result.all_messages()

        # ------------------------------------------------------------------
        # MODEL PROGRESS
        # ------------------------------------------------------------------

        self._trace_model_progress(
            result.new_messages(),
            run_steps_before,
        )

        # ------------------------------------------------------------------
        # STRUCTURED OUTPUT
        # ------------------------------------------------------------------

        output = result.output

        if isinstance(
            output,
            FlagFound,
        ):
            self._flag = (
                output.flag.strip()
            )

            self._findings = (
                f"Flag found via "
                f"{output.method}: "
                f"{self._flag}"
            )

            logger.info(
                "[%s] FLAG FOUND: %s",
                self.agent_name,
                self._flag,
            )

            # Local testing mode.
            if self.deps.no_submit:
                self._confirmed = True
                if state is not None:
                    mark_confirmed_flag(
                        state,
                        self._flag,
                    )

        # ------------------------------------------------------------------
        # CTFD CONFIRMATION
        # ------------------------------------------------------------------

        if self.deps.confirmed_flag:
            self._confirmed = True

            self._flag = (
                self._flag
                or self.deps.confirmed_flag
            )

        # ------------------------------------------------------------------
        # RUN STATISTICS
        # ------------------------------------------------------------------

        run_steps = (
            self._step_count[0]
            - run_steps_before
        )

        # ------------------------------------------------------------------
        # SUCCESS
        # ------------------------------------------------------------------

        if (
            self._confirmed
            and self._flag
        ):
            return self._result(
                FLAG_FOUND,
                run_steps=run_steps,
            )

        # ------------------------------------------------------------------
        # NO FLAG
        # ------------------------------------------------------------------

        self._findings = (
            self._findings
            or "No flag found in this attempt."
        )

        logger.info(
            "[%s] Run %d finished without flag "
            "(%d tool steps, %.1fs)",
            self.agent_name,
            self._run_number,
            run_steps,
            duration,
        )

        return self._result(
            GAVE_UP,
            run_steps=run_steps,
        )

    # ========================================================================
    # MODEL PROGRESS
    # ========================================================================

    def _trace_model_progress(
        self,
        messages: list[Any],
        run_steps_before: int,
    ) -> None:
        """Display model reasoning/progress summaries."""

        current_step = run_steps_before

        for msg in messages:

            if not isinstance(
                msg,
                ModelResponse,
            ):
                continue

            text_parts = [
                part.content
                for part in msg.parts
                if isinstance(
                    part,
                    TextPart,
                )
            ]

            text = " ".join(
                str(part)
                for part in text_parts
                if part is not None
            ).strip()

            if not text:
                continue

            msg_usage = msg.usage

            self.tracer.model_response(
                text[:1000],
                current_step,
                input_tokens=(
                    msg_usage.input_tokens
                    if msg_usage
                    else 0
                ),
                output_tokens=(
                    msg_usage.output_tokens
                    if msg_usage
                    else 0
                ),
            )

            logger.info(
                "[%s] DeepSeek progress: %s",
                self.agent_name,
                _compact_text(
                    text,
                    MAX_PROGRESS_TEXT,
                ),
            )

            self.tracer.event(
                "progress",
                text=text[:1000],
                step=current_step,
            )

    # ========================================================================
    # BUMP / RESUME
    # ========================================================================

    def bump(
        self,
        insights: str,
    ) -> None:
        """Start a fresh compact reasoning context."""

        # Never bump after a confirmed flag.

        if self.deps.confirmed_flag:

            self._flag = (
                self.deps.confirmed_flag
            )

            self._confirmed = True

            logger.info(
                "[%s] Bump skipped — flag already confirmed.",
                self.agent_name,
            )

            return

        insights = (
            insights
            or "No sibling insights available."
        )[:MAX_SIBLING_INSIGHTS]

        instruction = (
            self.deps.instruction.strip()
        )

        instruction_text = ""

        if instruction:
            instruction_text = (
                "\n\n"
                "Original operator instruction:\n"
                f"{instruction}\n"
            )

        if self.deps.investigation is not None:
            self.deps.investigation.retry_count += 1

        self._messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=(
                            "COMPACT RETRY — do not restart "
                            "reconnaissance.\n\n"
                            f"{compact_summary(self.deps.investigation) if self.deps.investigation else 'No investigation state.'}\n\n"
                            "Useful findings from other "
                            "solvers:\n\n"
                            f"{insights}"
                            f"{instruction_text}\n"
                            "Continue from CURRENT OBJECTIVE. "
                            "Do not repeat completed actions. "
                            "Do not inspect unrelated distfiles "
                            "unless new evidence requires it. "
                            "Use tools immediately and verify "
                            "any candidate flag."
                        )
                    )
                ],
            )
        ]

        self.loop_detector.reset()

        self.tracer.event(
            "bump",
            insights=insights[:500],
            run_number=self._run_number,
        )

        logger.info(
            "[%s] Starting compact new attempt",
            self.agent_name,
        )

    # ========================================================================
    # RESULT
    # ========================================================================

    def _result(
        self,
        status: str,
        run_steps: int | None = None,
        run_cost: float | None = None,
    ) -> SolverResult:
        """Build solver result."""

        agent_usage = (
            self.cost_tracker.by_agent.get(
                self.agent_name
            )
        )

        cost = (
            agent_usage.cost_usd
            if agent_usage
            else 0.0
        )

        self.tracer.event(
            "finish",
            status=status,
            flag=self._flag,
            confirmed=self._confirmed,
            cost_usd=round(
                cost,
                4,
            ),
            run_number=self._run_number,
        )

        return SolverResult(
            flag=self._flag,
            status=status,
            findings_summary=(
                self._findings[:2000]
            ),
            step_count=(
                run_steps
                if run_steps is not None
                else self._step_count[0]
            ),
            cost_usd=(
                run_cost
                if run_cost is not None
                else cost
            ),
            log_path=self.tracer.path,
        )

    # ========================================================================
    # STOP
    # ========================================================================

    async def stop(self) -> None:
        """Stop tracing and owned sandbox."""

        self.tracer.event(
            "stop",
            step_count=self._step_count[0],
        )

        self.tracer.close()

        if (
            self._owns_sandbox
            and self.sandbox
        ):
            await self.sandbox.stop()
