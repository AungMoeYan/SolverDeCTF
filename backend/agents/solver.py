"""Per-model solver agent — one model, one container, one challenge."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.deps import SolverDeps
from backend.investigation import (
    DEFAULT_REQUEST_BUDGET,
    InvestigationState,
    build_state_context,
    classify_challenge,
    detect_flag_in_text,
    extract_target,
    is_readonly_action,
    normalize_action,
)
from backend.loop_detect import LOOP_WARNING_MESSAGE, LoopDetector
from backend.models import (
    model_id_from_spec,
    provider_from_spec,
    resolve_model,
    resolve_model_settings,
    supports_vision,
)
from backend.output_types import FlagFound
from backend.prompts import ChallengeMeta, build_prompt, list_distfiles
from backend.sandbox import DockerSandbox
from backend.solver_base import (
    CANCELLED,
    CORRECT_MARKERS,
    ERROR,
    FLAG_FOUND,
    GAVE_UP,
    SolverResult,
)
from backend.tools.flag import submit_flag
from backend.tools.sandbox import (
    bash,
    check_findings,
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


# ---------------------------------------------------------------------------
# CLI / live execution helpers
# ---------------------------------------------------------------------------


def _shorten(value: Any, limit: int = 300) -> str:
    """Create a compact representation for live CLI output."""

    if value is None:
        return ""

    text = str(value).replace("\n", " ").strip()

    if len(text) > limit:
        return text[:limit] + "..."

    return text


def _format_tool_args(tool_args: dict[str, Any], limit: int = 500) -> str:
    """Format tool arguments for readable live CLI output."""

    if not tool_args:
        return ""

    parts: list[str] = []

    for key, value in tool_args.items():
        value_text = _shorten(value, 220)
        parts.append(f"{key}={value_text!r}")

    text = ", ".join(parts)

    if len(text) > limit:
        text = text[:limit] + "..."

    return text


# ---------------------------------------------------------------------------
# TracingToolset
# ---------------------------------------------------------------------------


@dataclass
class TracingToolset(WrapperToolset[SolverDeps]):
    """Wrap a toolset with tracing, loop detection and live investigation state."""

    tracer: SolverTracer = field(repr=False)
    loop_detector: LoopDetector = field(repr=False)
    step_counter: list[int] = field(repr=False)

    state: InvestigationState = field(
        default_factory=InvestigationState
    )

    request_budget: int = DEFAULT_REQUEST_BUDGET

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[SolverDeps],
        tool: ToolsetTool[SolverDeps],
    ) -> Any:
        """Execute a tool while exposing live solver activity."""

        self.step_counter[0] += 1
        step = self.step_counter[0]

        # ---------------------------------------------------------------
        # LIVE STEP: before tool execution
        # ---------------------------------------------------------------

        args_display = _format_tool_args(tool_args)

        if args_display:
            logger.info(
                f"[step {step}] → {name}({args_display})"
            )
        else:
            logger.info(
                f"[step {step}] → {name}()"
            )

        self.tracer.tool_call(
            name,
            tool_args,
            step,
        )

        # ---------------------------------------------------------------
        # Action deduplication
        # ---------------------------------------------------------------

        normalized = normalize_action(
            name,
            tool_args,
        )

        if (
            name != "submit_flag"
            and is_readonly_action(name, tool_args)
        ):
            seen, prior_summary = self.state.has_action(
                normalized
            )

            if seen:
                logger.info(
                    f"[step {step}] ↻ DEDUP: {name}"
                )

                self.tracer.event(
                    "dedup",
                    tool=name,
                    normalized=normalized,
                    step=step,
                )

                return (
                    "[DEDUP] You already performed this "
                    "equivalent action. "
                    f"Prior result: {prior_summary[:200]}\n\n"
                    "Try a DIFFERENT approach or move to your "
                    "next objective."
                )

        # ---------------------------------------------------------------
        # Loop detection
        # ---------------------------------------------------------------

        loop_status = self.loop_detector.check(
            name,
            tool_args,
        )

        if loop_status == "break":
            logger.warning(
                f"[step {step}] LOOP BREAK: {name}"
            )

            self.tracer.event(
                "loop_break",
                tool=name,
                step=step,
            )

            return LOOP_WARNING_MESSAGE

        if loop_status == "warn":
            logger.warning(
                f"[step {step}] LOOP WARNING: {name}"
            )

        # ---------------------------------------------------------------
        # Execute actual tool
        # ---------------------------------------------------------------

        try:
            result = await self.wrapped.call_tool(
                name,
                tool_args,
                ctx,
                tool,
            )
        except Exception as exc:
            logger.error(
                f"[step {step}] ✗ {name}: {exc}"
            )

            self.tracer.event(
                "tool_exception",
                tool=name,
                step=step,
                error=str(exc),
            )

            raise

        result_str = (
            str(result)
            if result is not None
            else ""
        )

        self.tracer.tool_result(
            name,
            result_str,
            step,
        )

        # ---------------------------------------------------------------
        # LIVE STEP: after tool execution
        # ---------------------------------------------------------------

        result_preview = _shorten(
            result_str,
            350,
        )

        if name == "submit_flag":
            if any(
                marker in result_str
                for marker in CORRECT_MARKERS
            ):
                logger.info(
                    f"[step {step}] ✓ FLAG CONFIRMED"
                )
            else:
                logger.info(
                    f"[step {step}] ← submit_flag: "
                    f"{result_preview}"
                )
        else:
            logger.info(
                f"[step {step}] ← {name}: "
                f"{result_preview}"
            )

        # ---------------------------------------------------------------
        # Record action
        # ---------------------------------------------------------------

        self.state.record_action(
            normalized,
            result_str,
        )

        self.state.tried.append(
            normalized
        )

        # ---------------------------------------------------------------
        # Flag detection
        # ---------------------------------------------------------------

        if (
            name != "submit_flag"
            and isinstance(result, str)
        ):
            flags = detect_flag_in_text(
                result
            )

            for flag in flags:
                if flag not in self.state.flag_candidates:
                    self.state.flag_candidates.append(
                        flag
                    )

                    self.tracer.event(
                        "flag_detected",
                        flag=flag,
                        tool=name,
                        step=step,
                    )

                    logger.info(
                        f"[step {step}] 🚩 FLAG CANDIDATE: "
                        f"{flag}"
                    )

        # ---------------------------------------------------------------
        # Progress detection
        # ---------------------------------------------------------------

        progress = self._categorize_progress(
            name,
            tool_args,
            result_str,
        )

        self.tracer.event(
            "progress",
            tool=name,
            progress=progress,
            step=step,
        )

        logger.info(
            f"[step {step}] state={progress}"
        )

        # ---------------------------------------------------------------
        # Update investigation state
        # ---------------------------------------------------------------

        self._update_state(
            name,
            tool_args,
            result_str,
            progress,
        )

        # ---------------------------------------------------------------
        # Loop warning
        # ---------------------------------------------------------------

        if loop_status == "warn":
            if isinstance(result, str):
                result = (
                    f"{result}\n\n"
                    f"{LOOP_WARNING_MESSAGE}"
                )

        # ---------------------------------------------------------------
        # Confirmed flag
        # ---------------------------------------------------------------

        if (
            name == "submit_flag"
            and any(
                marker in result_str
                for marker in CORRECT_MARKERS
            )
        ):
            self.tracer.event(
                "flag_confirmed",
                tool=name,
                step=step,
            )

        # ---------------------------------------------------------------
        # State / budget context
        # ---------------------------------------------------------------

        if (
            step % 3 == 0
            and isinstance(result, str)
        ):
            remaining = max(
                0,
                self.request_budget
                - self.step_counter[0],
            )

            # Flag candidates
            if self.state.flag_candidates:
                result += (
                    "\n\n**FLAG CANDIDATE(S)**: "
                    f"{', '.join(self.state.flag_candidates)}\n"
                    "Call `submit_flag` with the most "
                    "likely candidate NOW."
                )

            # Budget warning
            elif remaining <= 5 and remaining > 0:
                result += (
                    "\n\n**BUDGET WARNING**: "
                    f"~{remaining} requests remaining. "
                    "Focus on: "
                    f"{self.state.objective or 'your current best lead'}."
                )

            # No-progress warning
            if self.state.no_progress_streak >= 3:
                result += (
                    "\n\n**NO PROGRESS** for multiple actions. "
                    "STOP repeating similar actions. "
                    "Try a fundamentally different approach."
                )

            # IMPORTANT:
            # Build the state context separately.
            # This avoids the broken nested f-string that caused:
            #
            # SyntaxError: f-string: expecting '}'
            state_context = build_state_context(
                self.state,
                remaining,
            )

            result += (
                "\n\n---\n"
                + state_context
            )

            self.tracer.event(
                "state_injected",
                step=step,
            )

        # ---------------------------------------------------------------
        # Sibling findings
        # ---------------------------------------------------------------

        if (
            step % 5 == 0
            and ctx.deps.message_bus
            and isinstance(result, str)
        ):
            from backend.tools.core import (
                do_check_findings,
            )

            findings_text = await do_check_findings(
                ctx.deps.message_bus,
                ctx.deps.model_spec,
            )

            if (
                findings_text
                and "No new findings" not in findings_text
            ):
                result = (
                    f"{result}\n\n---\n"
                    f"{findings_text}"
                )

                self.tracer.event(
                    "findings_injected",
                    step=step,
                )

                logger.info(
                    f"[step {step}] ← sibling findings received"
                )

        return result

    # -------------------------------------------------------------------
    # Progress detection
    # -------------------------------------------------------------------

    def _categorize_progress(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
    ) -> str:
        """Categorize a tool result."""

        if not result or result in (
            "(no output)",
            "",
        ):
            return "NO_PROGRESS"

        error_indicators = [
            "error",
            "failed",
            "denied",
            "forbidden",
            "403",
            "404",
            "500",
            "command not found",
            "no such file",
            "permission denied",
            "connection refused",
            "timeout",
        ]

        result_lower = result.lower()

        error_count = sum(
            1
            for error in error_indicators
            if error in result_lower
        )

        if (
            error_count >= 2
            and len(result) < 500
        ):
            return "FAILURE"

        if (
            error_count >= 1
            and len(result) < 100
        ):
            return "FAILURE"

        has_flag = bool(
            detect_flag_in_text(result)
        )

        if has_flag:
            return "NEW_INFORMATION"

        has_new_urls = bool(
            re.findall(
                r'https?://[^\s<>"\']+',
                result,
            )
        )

        has_new_endpoints = bool(
            re.findall(
                r'/(?:api|admin|upload|flag|debug|backup|config|\.env|\.git)',
                result_lower,
            )
        )

        if (
            has_new_urls
            or has_new_endpoints
        ):
            return "NEW_INFORMATION"

        if len(result) > 100:
            for discovery in self.state.discoveries:
                if (
                    discovery.lower()
                    in result_lower
                    and len(result) < 300
                ):
                    return "CONFIRMATION"

            return "NEW_INFORMATION"

        return "NO_PROGRESS"

    # -------------------------------------------------------------------
    # Investigation state
    # -------------------------------------------------------------------

    def _update_state(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        progress: str,
    ) -> None:
        """Update investigation state from tool output."""

        if progress == "NEW_INFORMATION":

            # -----------------------------------------------------------
            # URLs
            # -----------------------------------------------------------

            for url in re.findall(
                r'https?://[^\s<>"\']+',
                result,
            ):
                self.state.add_endpoint(url)

            # -----------------------------------------------------------
            # Technologies
            # -----------------------------------------------------------

            tech_patterns = {
                "PHP": r"\bPHP\b",
                "Python": r"\bPython\b",
                "Flask": r"\bFlask\b",
                "Django": r"\bDjango\b",
                "Nginx": r"\bNginx\b",
                "Apache": r"\bApache\b",
                "MySQL": r"\bMySQL\b",
                "SQLite": r"\bSQLite\b",
                "RSA": r"\bRSA\b",
                "AES": r"\bAES\b",
                "Docker": r"\bDocker\b",
            }

            for tech, pattern in tech_patterns.items():
                if (
                    re.search(
                        pattern,
                        result,
                    )
                    and tech not in self.state.technologies
                ):
                    self.state.technologies.append(
                        tech
                    )

                    logger.info(
                        f"[state] technology={tech}"
                    )

            # -----------------------------------------------------------
            # Upload functionality
            # -----------------------------------------------------------

            if any(
                keyword in result.lower()
                for keyword in (
                    "upload",
                    "uploaded",
                    "file uploaded",
                )
            ):
                self.state.add_discovery(
                    "Upload functionality detected"
                )

                logger.info(
                    "[state] discovery=Upload functionality detected"
                )

            # -----------------------------------------------------------
            # SQL errors
            # -----------------------------------------------------------

            if (
                "sql" in result.lower()
                and "error" in result.lower()
            ):
                self.state.add_discovery(
                    "SQL error detected — possible SQLi"
                )

                logger.info(
                    "[state] discovery=Possible SQLi"
                )

        elif progress == "FAILURE":

            normalized = normalize_action(
                tool_name,
                args,
            )

            if normalized not in self.state.failed:
                self.state.failed.append(
                    normalized
                )

        elif progress == "NO_PROGRESS":

            self.state.increment_no_progress()

        # ---------------------------------------------------------------
        # Uploaded files
        # ---------------------------------------------------------------

        if tool_name == "write_file":
            path = args.get(
                "path",
                "",
            )

            if (
                path
                and path not in self.state.uploaded_files
            ):
                self.state.uploaded_files.append(
                    path
                )

        # ---------------------------------------------------------------
        # Invalidate read cache after state changes
        # ---------------------------------------------------------------

        if tool_name == "write_file":
            self.state._action_cache.clear()

        elif (
            tool_name == "bash"
            and not is_readonly_action(
                tool_name,
                args,
            )
        ):
            self.state._action_cache.clear()


# ---------------------------------------------------------------------------
# Toolset builder
# ---------------------------------------------------------------------------


def _build_toolset(
    deps: SolverDeps,
) -> FunctionToolset[SolverDeps]:
    """Build the raw toolset for a solver agent."""

    tools = [
        bash,
        read_file,
        write_file,
        list_files,
        submit_flag,
        web_fetch,
        webhook_create,
        webhook_get_requests,
        check_findings,
        notify_coordinator,
    ]

    if deps.use_vision:
        tools.append(
            view_image
        )

    return FunctionToolset(
        tools=tools,
        max_retries=4,
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class Solver:
    """A single solver: one model, one container, one challenge."""

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

        self.deps = SolverDeps(
            sandbox=self.sandbox,
            ctfd=ctfd,
            challenge_dir=challenge_dir,
            challenge_name=meta.name,
            workspace_dir="",
            use_vision=self.use_vision,
            cost_tracker=cost_tracker,
        )

        self.loop_detector = LoopDetector()

        self.tracer = SolverTracer(
            meta.name,
            self.model_id,
        )

        self.agent_name = (
            f"{meta.name}/{self.model_id}"
        )

        self._agent: Agent[
            SolverDeps,
            FlagFound,
        ] | None = None

        self._messages: list = []

        # Shared mutable counter used by TracingToolset.
        self._step_count = [0]

        self._flag: str | None = None
        self._confirmed = False
        self._findings = ""

        self.investigation = (
            InvestigationState()
        )

        self.request_budget = (
            DEFAULT_REQUEST_BUDGET
        )

    # -------------------------------------------------------------------
    # Start
    # -------------------------------------------------------------------

    async def start(self) -> None:
        """Start sandbox and construct the agent."""

        if not self.sandbox._container:
            await self.sandbox.start()

        self.deps.workspace_dir = (
            self.sandbox.workspace_dir
        )

        arch_result = await self.sandbox.exec(
            "uname -m",
            timeout_s=10,
        )

        container_arch = (
            arch_result.stdout.strip()
            or "unknown"
        )

        distfile_names = list_distfiles(
            self.challenge_dir
        )

        # ---------------------------------------------------------------
        # Challenge classification
        # ---------------------------------------------------------------

        category = classify_challenge(
            self.meta,
            distfile_names,
        )

        self.investigation.category = (
            category
        )

        self.investigation.target = (
            extract_target(
                self.meta.connection_info
            )
        )

        logger.info(
            f"[state] category={category} "
            f"target={self.investigation.target}"
        )

        system_prompt = build_prompt(
            self.meta,
            distfile_names,
            container_arch=container_arch,
            category_hint=category,
        )

        model = resolve_model(
            self.model_spec,
            self.settings,
        )

        model_settings = (
            resolve_model_settings(
                self.model_spec
            )
        )

        raw_toolset = _build_toolset(
            self.deps
        )

        toolset = TracingToolset(
            wrapped=raw_toolset,
            tracer=self.tracer,
            loop_detector=self.loop_detector,
            step_counter=self._step_count,
            state=self.investigation,
            request_budget=self.request_budget,
        )

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
        )

        logger.info(
            f"[{self.agent_name}] "
            f"Solver started "
            f"(category={category})"
        )

    # -------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------

    async def run_until_done_or_gave_up(
        self,
    ) -> SolverResult:
        """Run until flag found, budget exhausted, or cancelled."""

        if not self._agent:
            await self.start()

        assert self._agent is not None

        from pydantic_ai.exceptions import (
            UsageLimitExceeded,
        )

        from pydantic_ai.usage import (
            UsageLimits,
        )

        t0 = time.monotonic()

        logger.info(
            "============================================================"
        )

        logger.info(
            f"[solver] Starting autonomous run "
            f"with budget={self.request_budget}"
        )

        logger.info(
            "============================================================"
        )

        try:

            result = await self._agent.run(
                (
                    "Solve this CTF challenge."
                    if not self._messages
                    else "Continue solving."
                ),
                deps=self.deps,
                message_history=(
                    self._messages
                    if self._messages
                    else None
                ),
                usage_limits=UsageLimits(
                    request_limit=self.request_budget
                ),
            )

            duration = (
                time.monotonic()
                - t0
            )

            usage = result.usage

            # -----------------------------------------------------------
            # Cost tracking
            # -----------------------------------------------------------

            self.cost_tracker.record(
                self.agent_name,
                usage,
                self.model_id,
                provider_spec=provider_from_spec(
                    self.model_spec
                ),
                duration_seconds=duration,
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

            self._messages = (
                result.all_messages()
            )

            # -----------------------------------------------------------
            # Trace model responses
            # -----------------------------------------------------------

            from pydantic_ai.messages import (
                ModelResponse,
                TextPart,
            )

            for msg in result.new_messages():

                if isinstance(
                    msg,
                    ModelResponse,
                ):

                    text_parts = [
                        part.content
                        for part in msg.parts
                        if isinstance(
                            part,
                            TextPart,
                        )
                    ]

                    text = " ".join(
                        text_parts
                    )

                    msg_usage = msg.usage

                    self.tracer.model_response(
                        text[:500],
                        self._step_count[0],
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

            # -----------------------------------------------------------
            # Structured output
            # -----------------------------------------------------------

            output = result.output

            if isinstance(
                output,
                FlagFound,
            ):

                self._flag = output.flag

                self._findings = (
                    f"Flag found via "
                    f"{output.method}: "
                    f"{self._flag}"
                )

                logger.info(
                    f"[solver] Model reported flag: "
                    f"{self._flag}"
                )

                if self.deps.no_submit:
                    self._confirmed = True

            # -----------------------------------------------------------
            # Confirmed by submit_flag
            # -----------------------------------------------------------

            if self.deps.confirmed_flag:

                self._confirmed = True

                self._flag = (
                    self._flag
                    or self.deps.confirmed_flag
                )

                logger.info(
                    f"[solver] Confirmed flag: "
                    f"{self.deps.confirmed_flag}"
                )

            # -----------------------------------------------------------
            # Investigation candidates
            # -----------------------------------------------------------

            if (
                not self._confirmed
                and self.investigation.flag_candidates
            ):

                best = (
                    self.investigation.flag_candidates[-1]
                )

                self._flag = best

                self._findings = (
                    "Flag candidate from "
                    f"investigation: {best}"
                )

                logger.info(
                    f"[solver] Candidate preserved: "
                    f"{best}"
                )

            # -----------------------------------------------------------
            # Final result
            # -----------------------------------------------------------

            if (
                self._confirmed
                and self._flag
            ):
                logger.info(
                    "[solver] FLAG FOUND"
                )

                return self._result(
                    FLAG_FOUND
                )

            logger.info(
                "[solver] Solver finished without "
                "confirmed flag"
            )

            return self._result(
                GAVE_UP
            )

        # ---------------------------------------------------------------
        # Usage limit
        # ---------------------------------------------------------------

        except UsageLimitExceeded as exc:

            logger.warning(
                f"[{self.agent_name}] "
                f"Budget exhausted: {exc}"
            )

            if self.deps.confirmed_flag:

                self._confirmed = True

                self._flag = (
                    self._flag
                    or self.deps.confirmed_flag
                )

            if (
                not self._confirmed
                and self.investigation.flag_candidates
            ):

                self._flag = (
                    self.investigation.flag_candidates[-1]
                )

                self._findings = (
                    "Flag candidate "
                    "(budget exhausted): "
                    f"{self._flag}"
                )

            elif not self._findings:

                self._findings = (
                    "Budget exhausted after "
                    f"{self._step_count[0]} "
                    "tool calls"
                )

            return self._result(
                GAVE_UP
            )

        # ---------------------------------------------------------------
        # Cancellation
        # ---------------------------------------------------------------

        except asyncio.CancelledError:

            logger.warning(
                f"[{self.agent_name}] "
                "Solver cancelled"
            )

            if self.deps.confirmed_flag:

                self._confirmed = True

                self._flag = (
                    self._flag
                    or self.deps.confirmed_flag
                )

            if self.investigation.flag_candidates:

                self._flag = (
                    self.investigation.flag_candidates[-1]
                )

                self._findings = (
                    "Flag candidate "
                    "(cancelled): "
                    f"{self._flag}"
                )

            return self._result(
                CANCELLED
            )

        # ---------------------------------------------------------------
        # Unexpected error
        # ---------------------------------------------------------------

        except Exception as exc:

            if self.deps.confirmed_flag:

                self._confirmed = True

                self._flag = (
                    self._flag
                    or self.deps.confirmed_flag
                )

            logger.error(
                f"[{self.agent_name}] "
                f"Error: {exc}",
                exc_info=True,
            )

            self._findings = (
                f"Error: {exc}"
            )

            self.tracer.event(
                "error",
                error=str(exc),
            )

            if self.investigation.flag_candidates:

                self._flag = (
                    self.investigation.flag_candidates[-1]
                )

                self._findings = (
                    "Flag candidate "
                    "(error): "
                    f"{self._flag}"
                )

            return self._result(
                ERROR
            )

    # -------------------------------------------------------------------
    # Bump
    # -------------------------------------------------------------------

    def bump(
        self,
        insights: str,
    ) -> None:
        """Inject insights from sibling agents."""

        state_context = build_state_context(
            self.investigation,
            max(
                0,
                self.request_budget
                - self._step_count[0],
            ),
        )

        bump_msg = ModelRequest(
            parts=[
                UserPromptPart(
                    content=(
                        "Your previous attempt did not "
                        "find the flag. Here are insights "
                        "from other agents working on the "
                        "same challenge:\n\n"
                        f"{insights}\n\n"
                        "Use these insights to try a "
                        "different approach. "
                        "Do NOT repeat what has already "
                        "been tried.\n\n"
                        f"{state_context}"
                    )
                )
            ]
        )

        self._messages.append(
            bump_msg
        )

        self.loop_detector.reset()

        self.tracer.event(
            "bump",
            insights=insights[:500],
        )

        logger.info(
            f"[{self.agent_name}] "
            "Bumped with sibling insights"
        )

    # -------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------

    def _result(
        self,
        status: str,
        run_steps: int | None = None,
        run_cost: float | None = None,
    ) -> SolverResult:

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

    # -------------------------------------------------------------------
    # Stop
    # -------------------------------------------------------------------

    async def stop(self) -> None:

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
