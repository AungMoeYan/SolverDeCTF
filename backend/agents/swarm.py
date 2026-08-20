"""ChallengeSwarm — parallel solvers racing on one challenge."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.agents.solver import Solver
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.message_bus import ChallengeMessageBus
from backend.models import DEFAULT_MODELS, provider_from_spec
from backend.prompts import ChallengeMeta
from backend.solver_base import (
    CANCELLED,
    ERROR,
    FLAG_FOUND,
    GAVE_UP,
    QUOTA_ERROR,
    SolverProtocol,
    SolverResult,
)

if TYPE_CHECKING:
    from backend.config import Settings

logger = logging.getLogger(__name__)


# ============================================================
# QUOTA FALLBACK
# ============================================================

QUOTA_FALLBACK: dict[str, str] = {
    "claude-sdk/claude-opus-4-6":
        "bedrock/us.anthropic.claude-opus-4-6-v1",

    "codex/gpt-5.4":
        "azure/gpt-5.4",

    "codex/gpt-5.4-mini":
        "azure/gpt-5.4-mini",

    "codex/gpt-5.3-codex-spark":
        "zen/gpt-5.3-codex-spark",
}


def _quota_fallback_spec(model_spec: str) -> str | None:
    return QUOTA_FALLBACK.get(model_spec)


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

# Number of times a solver can be continued after giving up.
#
# Initial attempt + 2 continuation attempts = 3 total runs.
MAX_BUMPS = 2

# Short delay to allow sibling agents to publish findings.
BUMP_DELAY_SECONDS = 3

# Pydantic AI request budget per solver run.
#
# DeepSeek needs multiple model/tool turns for CTF solving.
REQUEST_LIMIT = 30


# ============================================================
# CHALLENGE SWARM
# ============================================================

@dataclass
class ChallengeSwarm:
    """Parallel solvers racing and cooperating on one challenge."""

    challenge_dir: str
    meta: ChallengeMeta
    ctfd: CTFdClient
    cost_tracker: CostTracker
    settings: Settings

    model_specs: list[str] = field(
        default_factory=lambda: list(DEFAULT_MODELS)
    )

    no_submit: bool = False

    coordinator_inbox: asyncio.Queue | None = None

    cancel_event: asyncio.Event = field(
        default_factory=asyncio.Event
    )

    solvers: dict[str, SolverProtocol] = field(
        default_factory=dict
    )

    findings: dict[str, str] = field(
        default_factory=dict
    )

    winner: SolverResult | None = None

    confirmed_flag: str | None = None

    _flag_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock
    )

    _submit_count: dict[str, int] = field(
        default_factory=dict
    )

    _submitted_flags: set[str] = field(
        default_factory=set
    )

    _last_submit_time: dict[str, float] = field(
        default_factory=dict
    )

    message_bus: ChallengeMessageBus = field(
        default_factory=ChallengeMessageBus
    )

    # ============================================================
    # CREATE SOLVER
    # ============================================================

    def _create_solver(self, model_spec: str):
        """Create the appropriate solver for the provider."""

        provider = provider_from_spec(model_spec)

        def _submit_fn(flag):
            return self.try_submit_flag(
                flag,
                model_spec,
            )

        notify = self._make_notify_fn(
            model_spec
        )

        # --------------------------------------------------------
        # Claude SDK
        # --------------------------------------------------------

        if provider == "claude-sdk":

            from backend.agents.claude_solver import ClaudeSolver

            return ClaudeSolver(
                model_spec=model_spec,
                challenge_dir=self.challenge_dir,
                meta=self.meta,
                ctfd=self.ctfd,
                cost_tracker=self.cost_tracker,
                settings=self.settings,
                cancel_event=self.cancel_event,
                no_submit=self.no_submit,
                submit_fn=_submit_fn,
                message_bus=self.message_bus,
                notify_coordinator=notify,
            )

        # --------------------------------------------------------
        # Codex
        # --------------------------------------------------------

        if provider == "codex":

            from backend.agents.codex_solver import CodexSolver

            return CodexSolver(
                model_spec=model_spec,
                challenge_dir=self.challenge_dir,
                meta=self.meta,
                ctfd=self.ctfd,
                cost_tracker=self.cost_tracker,
                settings=self.settings,
                cancel_event=self.cancel_event,
                no_submit=self.no_submit,
                submit_fn=_submit_fn,
                message_bus=self.message_bus,
                notify_coordinator=notify,
            )

        # --------------------------------------------------------
        # Pydantic AI providers
        #
        # bedrock
        # azure
        # zen
        # google
        # --------------------------------------------------------

        return self._create_pydantic_solver(
            model_spec
        )

    # ============================================================
    # COORDINATOR MESSAGE
    # ============================================================

    def _make_notify_fn(self, model_spec: str):

        async def _notify(message: str) -> None:

            if self.coordinator_inbox:

                self.coordinator_inbox.put_nowait(
                    f"[{self.meta.name}/{model_spec}] {message}"
                )

        return _notify

    # ============================================================
    # PYDANTIC AI SOLVER
    # ============================================================

    def _create_pydantic_solver(
        self,
        model_spec: str,
        sandbox=None,
        owns_sandbox: bool | None = None,
    ) -> Solver:

        solver = Solver(
            model_spec=model_spec,
            challenge_dir=self.challenge_dir,
            meta=self.meta,
            ctfd=self.ctfd,
            cost_tracker=self.cost_tracker,
            settings=self.settings,
            cancel_event=self.cancel_event,
            sandbox=sandbox,
            owns_sandbox=owns_sandbox,
        )

        solver.deps.message_bus = self.message_bus
        solver.deps.model_spec = model_spec
        solver.deps.no_submit = self.no_submit

        solver.deps.submit_fn = (
            lambda flag:
            self.try_submit_flag(
                flag,
                model_spec,
            )
        )

        solver.deps.notify_coordinator = (
            self._make_notify_fn(
                model_spec
            )
        )

        return solver

    # ============================================================
    # SIBLING FINDINGS
    # ============================================================

    def _gather_sibling_insights(
        self,
        exclude_model: str,
    ) -> str:

        parts: list[str] = []

        for model, finding in self.findings.items():

            if model == exclude_model:
                continue

            if not finding:
                continue

            # Keep context small.
            finding = finding[:1200]

            parts.append(
                f"[{model}]\n{finding}"
            )

        if not parts:

            return (
                "No sibling insights are available yet."
            )

        return "\n\n".join(parts)

    # ============================================================
    # FLAG SUBMISSION
    # ============================================================

    SUBMISSION_COOLDOWNS = [
        0,
        10,
        30,
        60,
        120,
    ]

    async def try_submit_flag(
        self,
        flag: str,
        model_spec: str,
    ) -> tuple[str, bool]:

        async with self._flag_lock:

            # ----------------------------------------------------
            # Already solved
            # ----------------------------------------------------

            if self.confirmed_flag:

                return (
                    "ALREADY SOLVED — "
                    f"flag already confirmed: "
                    f"{self.confirmed_flag}",
                    True,
                )

            normalized = flag.strip()

            if not normalized:

                return (
                    "INVALID — empty flag.",
                    False,
                )

            # ----------------------------------------------------
            # Duplicate flag
            # ----------------------------------------------------

            if normalized in self._submitted_flags:

                return (
                    "INCORRECT — "
                    "this exact flag was already tried.",
                    False,
                )

            # ----------------------------------------------------
            # Submission cooldown
            # ----------------------------------------------------

            wrong_count = self._submit_count.get(
                model_spec,
                0,
            )

            cooldown_idx = min(
                wrong_count,
                len(self.SUBMISSION_COOLDOWNS) - 1,
            )

            cooldown = self.SUBMISSION_COOLDOWNS[
                cooldown_idx
            ]

            if cooldown:

                last_time = self._last_submit_time.get(
                    model_spec,
                    0,
                )

                elapsed = (
                    time.monotonic()
                    - last_time
                )

                if elapsed < cooldown:

                    remaining = max(
                        1,
                        int(cooldown - elapsed),
                    )

                    return (
                        f"COOLDOWN — wait {remaining}s "
                        "before submitting again.",
                        False,
                    )

            self._submitted_flags.add(
                normalized
            )

        # ----------------------------------------------------
        # Dry run
        # ----------------------------------------------------

        if self.no_submit:

            logger.info(
                "[%s/%s] Dry-run candidate: %s",
                self.meta.name,
                model_spec,
                normalized,
            )

            return (
                f"DRY RUN — candidate flag: {normalized}",
                False,
            )

        # ----------------------------------------------------
        # Actual CTFd submission
        # ----------------------------------------------------

        from backend.tools.core import do_submit_flag

        try:
            display, confirmed = await do_submit_flag(
                self.ctfd,
                self.meta.name,
                normalized,
            )
        except Exception as exc:
            logger.warning(
                "[%s/%s] Flag submission failed: %s",
                self.meta.name,
                model_spec,
                exc,
            )

            return (
                f"SUBMISSION ERROR — {exc}",
                False,
            )

        # Only mark the flag as submitted after the request
        # actually reached CTFd.
        self._submitted_flags.add(normalized)

        if confirmed:

            self.confirmed_flag = normalized

            logger.info(
                "[%s] FLAG CONFIRMED by %s: %s",
                self.meta.name,
                model_spec,
                normalized,
            )

            self.cancel_event.set()

            return (
                display,
                True,
            )

        # ----------------------------------------------------
        # Incorrect submission
        # ----------------------------------------------------

        self._submit_count[model_spec] = (
            wrong_count + 1
        )

        self._last_submit_time[model_spec] = (
            time.monotonic()
        )

        return (
            display,
            False,
        )

    # ============================================================
    # RUN ONE SOLVER
    # ============================================================

    async def _run_solver(
        self,
        model_spec: str,
    ) -> SolverResult | None:

        solver = self._create_solver(
            model_spec
        )

        self.solvers[model_spec] = solver

        try:

            result, final_solver = (
                await self._run_solver_loop(
                    solver,
                    model_spec,
                )
            )

            solver = final_solver

            return result

        except Exception as exc:

            logger.error(
                "[%s/%s] Fatal: %s",
                self.meta.name,
                model_spec,
                exc,
                exc_info=True,
            )

            return None

        finally:

            await solver.stop()

    # ============================================================
    # SOLVER LOOP
    # ============================================================

    async def _run_solver_loop(
        self,
        solver,
        model_spec: str,
    ) -> tuple[SolverResult, SolverProtocol]:

        bump_count = 0
        consecutive_errors = 0

        result = SolverResult(
            flag=None,
            status=CANCELLED,
            findings_summary="",
            step_count=0,
            cost_usd=0.0,
            log_path="",
        )

        # --------------------------------------------------------
        # Start solver
        # --------------------------------------------------------

        await solver.start()

        while not self.cancel_event.is_set():

            # ----------------------------------------------------
            # Run model
            # ----------------------------------------------------

            result = (
                await solver.run_until_done_or_gave_up()
            )

            # ----------------------------------------------------
            # Store useful findings
            # ----------------------------------------------------

            if (
                result.status
                not in (
                    ERROR,
                    QUOTA_ERROR,
                )
                and result.step_count > 0
                and result.findings_summary
                and not result.findings_summary.startswith(
                    (
                        "Error:",
                        "Turn failed:",
                    )
                )
            ):

                self.findings[model_spec] = (
                    result.findings_summary[:1500]
                )

                await self.message_bus.post(
                    model_spec,
                    result.findings_summary[:800],
                )

            # ----------------------------------------------------
            # FLAG FOUND
            # ----------------------------------------------------

            if result.status == FLAG_FOUND:

                self.cancel_event.set()

                self.winner = result

                logger.info(
                    "[%s] Flag found by %s: %s",
                    self.meta.name,
                    model_spec,
                    result.flag,
                )

                return result, solver

            # ----------------------------------------------------
            # CANCELLED
            # ----------------------------------------------------

            if result.status == CANCELLED:
                break

            # ----------------------------------------------------
            # QUOTA ERROR
            # ----------------------------------------------------

            if result.status == QUOTA_ERROR:

                fallback_spec = (
                    _quota_fallback_spec(
                        model_spec
                    )
                )

                if fallback_spec:

                    logger.warning(
                        "[%s/%s] Quota exhausted — "
                        "falling back to %s",
                        self.meta.name,
                        model_spec,
                        fallback_spec,
                    )

                    existing_sandbox = (
                        solver.sandbox
                    )

                    # Prevent old solver from deleting
                    # the shared sandbox.
                    solver.sandbox = None

                    await solver.stop()

                    solver = (
                        self._create_pydantic_solver(
                            fallback_spec,
                            sandbox=existing_sandbox,
                            owns_sandbox=True,
                        )
                    )

                    self.solvers[model_spec] = solver

                    await solver.start()

                    bump_count = 0
                    consecutive_errors = 0

                    continue

                break

            # ----------------------------------------------------
            # GAVE UP / ERROR
            # ----------------------------------------------------

            if result.status in (
                GAVE_UP,
                ERROR,
            ):

                # Broken solver.
                if (
                    result.step_count == 0
                    and result.cost_usd == 0
                ):

                    logger.warning(
                        "[%s/%s] Broken solver — "
                        "zero steps.",
                        self.meta.name,
                        model_spec,
                    )

                    break

                # ------------------------------------------------
                # Error tracking
                # ------------------------------------------------

                if result.status == ERROR:

                    consecutive_errors += 1

                    if consecutive_errors >= 3:

                        logger.warning(
                            "[%s/%s] Three consecutive errors.",
                            self.meta.name,
                            model_spec,
                        )

                        break

                else:

                    consecutive_errors = 0

                # ------------------------------------------------
                # Maximum continuation rounds
                # ------------------------------------------------

                if bump_count >= MAX_BUMPS:

                    logger.info(
                        "[%s/%s] Maximum continuation "
                        "rounds reached.",
                        self.meta.name,
                        model_spec,
                    )

                    break

                bump_count += 1

                # ------------------------------------------------
                # Give siblings time to publish findings
                # ------------------------------------------------

                try:

                    await asyncio.wait_for(
                        self.cancel_event.wait(),
                        timeout=BUMP_DELAY_SECONDS,
                    )

                    break

                except TimeoutError:
                    pass

                # ------------------------------------------------
                # Get sibling intelligence
                # ------------------------------------------------

                insights = (
                    self._gather_sibling_insights(
                        model_spec
                    )
                )

                # ------------------------------------------------
                # Resume with new information
                # ------------------------------------------------

                solver.bump(
                    insights
                )

                logger.info(
                    "[%s/%s] Bumped %d/%d",
                    self.meta.name,
                    model_spec,
                    bump_count,
                    MAX_BUMPS,
                )

                continue

        return result, solver

    # ============================================================
    # RUN SWARM
    # ============================================================

    async def run(
        self,
    ) -> SolverResult | None:

        if not self.model_specs:

            logger.error(
                "[%s] No models configured.",
                self.meta.name,
            )

            return None

        logger.info(
            "[%s] Starting swarm: %s",
            self.meta.name,
            ", ".join(self.model_specs),
        )

        # --------------------------------------------------------
        # Start all models concurrently.
        # --------------------------------------------------------

        tasks = [
            asyncio.create_task(
                self._run_solver(spec),
                name=f"solver-{spec}",
            )
            for spec in self.model_specs
        ]

        try:

            while tasks:

                done, pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # ------------------------------------------------
                # Check completed tasks
                # ------------------------------------------------

                for task in done:

                    try:

                        result = task.result()

                    except asyncio.CancelledError:

                        continue

                    except Exception as exc:

                        logger.error(
                            "[%s] Solver failed: %s",
                            self.meta.name,
                            exc,
                            exc_info=True,
                        )

                        continue

                    if not result:
                        continue

                    # --------------------------------------------
                    # Winner found
                    # --------------------------------------------

                    if result.status == FLAG_FOUND:

                        self.cancel_event.set()

                        self.winner = result

                        logger.info(
                            "[%s] Winner: %s",
                            self.meta.name,
                            result.flag,
                        )

                        # Stop every remaining solver.
                        for pending_task in pending:
                            pending_task.cancel()

                        await asyncio.gather(
                            *pending,
                            return_exceptions=True,
                        )

                        return result

                tasks = list(pending)

            # ----------------------------------------------------
            # Nobody found flag.
            # ----------------------------------------------------

            self.cancel_event.set()

            return self.winner

        except asyncio.CancelledError:

            self.cancel_event.set()

            for task in tasks:
                task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            raise

        except Exception as exc:

            logger.error(
                "[%s] Swarm error: %s",
                self.meta.name,
                exc,
                exc_info=True,
            )

            self.cancel_event.set()

            for task in tasks:
                task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True,
            )

            return None

    # ============================================================
    # KILL
    # ============================================================

    def kill(self) -> None:
        """Cancel all agents."""

        self.cancel_event.set()

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(self) -> dict:

        agents = {}

        for spec in self.model_specs:

            if (
                self.winner
                and self.winner.flag
            ):

                status = "won"

            elif (
                spec in self.solvers
                and not self.cancel_event.is_set()
            ):

                status = "running"

            else:

                status = "finished"

            agents[spec] = {
                "findings": self.findings.get(
                    spec,
                    "",
                ),
                "status": status,
            }

        return {
            "challenge": self.meta.name,
            "cancelled": self.cancel_event.is_set(),
            "winner": (
                self.winner.flag
                if self.winner
                else None
            ),
            "agents": agents,
        }
