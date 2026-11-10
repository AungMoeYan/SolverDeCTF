"""CTF Agent CLI entry point.

Architecture:

    ONE CTF
        |
        v
    ONE Solver
        |
        v
    ONE LLM
        |
        v
    ONE Docker sandbox
        |
        v
    FLAG / GIVE UP

The solver itself owns the agentic tool loop.  The CLI is responsible only
for collecting input, constructing dependencies, starting the solver, and
displaying the result.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from backend.config import Settings
from backend.models import DEFAULT_MODELS

console = Console()


# ============================================================================
# LOGGING
# ============================================================================


def _setup_logging(verbose: bool = False) -> None:
    """Configure application logging."""

    level = logging.DEBUG if verbose else logging.INFO

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiodocker").setLevel(logging.WARNING)

    handler = logging.StreamHandler()

    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%X",
        )
    )

    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,
    )


# ============================================================================
# INTERACTIVE INPUT
# ============================================================================


def _read_multiline(prompt: str) -> str:
    """Read multiline operator input.

    Input ends when the user enters an empty line.
    """

    console.print(
        f"\n[bold cyan]{prompt}[/bold cyan]"
    )

    console.print(
        "[dim]Enter your text. "
        "Press ENTER twice when finished.[/dim]"
    )

    lines: list[str] = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        if not line.strip():
            break

        lines.append(line)

    return "\n".join(lines).strip()


def _interactive_session() -> tuple[str, str, list[str]]:
    """Collect challenge information from the operator."""

    console.print()

    console.print(
        Panel(
            "[bold cyan]CTF Agent Interactive Mode[/bold cyan]\n\n"
            "Describe the challenge, provide any useful hint, "
            "and optionally provide local evidence files/directories.",
            border_style="cyan",
        )
    )

    description = _read_multiline(
        "Challenge description"
    )

    if not description:
        console.print(
            "[red]Challenge description cannot be empty.[/red]"
        )
        raise click.Abort()

    hint = _read_multiline(
        "Hint (optional)"
    )

    console.print(
        "\n[bold cyan]Evidence paths (optional)[/bold cyan]"
    )

    console.print(
        "[dim]Examples:[/dim]"
    )

    console.print(
        "[dim]  /home/kali/Downloads/disk.dd[/dim]"
    )

    console.print(
        "[dim]  /home/kali/Downloads/evidence/[/dim]"
    )

    console.print(
        "[dim]Enter one path per line. "
        "Press ENTER twice when finished.[/dim]"
    )

    evidence: list[str] = []

    while True:
        try:
            line = input()
        except EOFError:
            break

        line = line.strip()

        if not line:
            break

        evidence.append(line)

    return description, hint, evidence


# ============================================================================
# CLI
# ============================================================================


@click.command()
@click.option(
    "--ctfd-url",
    default=None,
    help="CTFd URL (overrides .env).",
)
@click.option(
    "--ctfd-token",
    default=None,
    help="CTFd API token (overrides .env).",
)
@click.option(
    "--image",
    default="ctf-sandbox",
    help="Docker sandbox image name.",
)
@click.option(
    "--model",
    "model_spec",
    default=None,
    help="Single LLM model specification.",
)
@click.option(
    "--challenge",
    default=None,
    help="Solve an existing challenge directory.",
)
@click.option(
    "--challenges-dir",
    default="challenges",
    help="Directory for challenge files.",
)
@click.option(
    "--instruction",
    default="",
    help="Extra instruction for the solver.",
)
@click.option(
    "--description",
    default="",
    help="Challenge description.",
)
@click.option(
    "--hint",
    default="",
    help="Challenge hint.",
)
@click.option(
    "--evidence",
    multiple=True,
    help="Host evidence file/directory.",
)
@click.option(
    "--no-submit",
    is_flag=True,
    help="Dry run — don't submit flags.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Verbose logging.",
)
def main(
    ctfd_url: str | None,
    ctfd_token: str | None,
    image: str,
    model_spec: str | None,
    challenge: str | None,
    challenges_dir: str,
    instruction: str,
    description: str,
    hint: str,
    evidence: tuple[str, ...],
    no_submit: bool,
    verbose: bool,
) -> None:
    """Run one CTF challenge with one autonomous solver."""

    _setup_logging(verbose)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    settings = Settings(
        sandbox_image=image,
    )

    if ctfd_url:
        settings.ctfd_url = ctfd_url

    if ctfd_token:
        settings.ctfd_token = ctfd_token

    # Preserve operator instruction.
    object.__setattr__(
        settings,
        "solver_instruction",
        instruction.strip(),
    )

    # ------------------------------------------------------------------
    # ONE MODEL ONLY
    # ------------------------------------------------------------------

    selected_model = (
        model_spec
        or DEFAULT_MODELS[0]
    )

    console.print(
        "\n[bold]CTF Agent[/bold]"
    )

    console.print(
        "  Architecture: [green]ONE CTF → ONE AGENT → ONE LLM[/green]"
    )

    console.print(
        f"  Model: [cyan]{selected_model}[/cyan]"
    )

    console.print(
        f"  Image: [cyan]{settings.sandbox_image}[/cyan]"
    )

    # ------------------------------------------------------------------
    # Interactive mode
    # ------------------------------------------------------------------

    if (
        challenge is None
        and not description.strip()
        and not instruction.strip()
    ):
        (
            description,
            hint,
            interactive_evidence,
        ) = _interactive_session()

        evidence = (
            *evidence,
            *interactive_evidence,
        )

    # ------------------------------------------------------------------
    # Display operator input
    # ------------------------------------------------------------------

    if description.strip():
        console.print(
            "\n[bold]Description:[/bold]"
        )
        console.print(
            description.strip()[:2000]
        )

    if hint.strip():
        console.print(
            "\n[bold]Hint:[/bold]"
        )
        console.print(
            hint.strip()[:2000]
        )

    if evidence:
        console.print(
            "\n[bold]Evidence:[/bold]"
        )

        for path in evidence:
            console.print(
                f"  [cyan]{path}[/cyan]"
            )

    console.print()

    # ------------------------------------------------------------------
    # Existing challenge
    # ------------------------------------------------------------------

    if challenge:
        asyncio.run(
            _run_single(
                settings=settings,
                challenge_dir=challenge,
                model_spec=selected_model,
                no_submit=no_submit,
                description=description,
                hint=hint,
                evidence=list(evidence),
            )
        )

        return

    # ------------------------------------------------------------------
    # Interactive challenge
    # ------------------------------------------------------------------

    asyncio.run(
        _run_interactive(
            settings=settings,
            model_spec=selected_model,
            no_submit=no_submit,
            description=description,
            hint=hint,
            evidence=list(evidence),
        )
    )


# ============================================================================
# COMMON SOLVER CONFIGURATION
# ============================================================================


def _configure_solver_context(
    settings: Settings,
    description: str,
    hint: str,
    evidence: list[str],
) -> None:
    """Make operator-provided information available to the Solver."""

    instruction_parts: list[str] = []

    if description.strip():
        instruction_parts.append(
            "Operator challenge description:\n"
            f"{description.strip()}"
        )

    if hint.strip():
        instruction_parts.append(
            "Operator hint:\n"
            f"{hint.strip()}"
        )

    if evidence:
        instruction_parts.append(
            "Operator-provided evidence paths:\n"
            + "\n".join(
                f"- {path}"
                for path in evidence
            )
        )

    object.__setattr__(
        settings,
        "solver_instruction",
        "\n\n".join(instruction_parts),
    )

    object.__setattr__(
        settings,
        "solver_description",
        description.strip(),
    )

    object.__setattr__(
        settings,
        "solver_hint",
        hint.strip(),
    )

    object.__setattr__(
        settings,
        "solver_evidence",
        evidence,
    )


# ============================================================================
# SINGLE CHALLENGE
# ============================================================================


async def _run_single(
    settings: Settings,
    challenge_dir: str,
    model_spec: str,
    no_submit: bool,
    description: str = "",
    hint: str = "",
    evidence: list[str] | None = None,
) -> None:
    """Solve exactly one challenge with exactly one Solver."""

    from backend.agents.solver import Solver
    from backend.cost_tracker import CostTracker
    from backend.ctfd import CTFdClient
    from backend.prompts import ChallengeMeta
    from backend.sandbox import (
        cleanup_orphan_containers,
        configure_semaphore,
    )
    from backend.solver_base import FLAG_FOUND

    evidence = evidence or []

    # ------------------------------------------------------------------
    # Solver context
    # ------------------------------------------------------------------

    _configure_solver_context(
        settings=settings,
        description=description,
        hint=hint,
        evidence=evidence,
    )

    # ------------------------------------------------------------------
    # Sandbox
    #
    # ONE challenge = ONE active sandbox.
    # ------------------------------------------------------------------

    configure_semaphore(1)

    await cleanup_orphan_containers()

    challenge_path = Path(
        challenge_dir
    )

    meta_path = (
        challenge_path
        / "metadata.yml"
    )

    if not meta_path.exists():
        console.print(
            f"[red]No metadata.yml found in "
            f"{challenge_dir}[/red]"
        )
        return

    meta = ChallengeMeta.from_yaml(
        meta_path
    )

    console.print(
        f"\n[bold]Challenge:[/bold] "
        f"{meta.name} "
        f"({meta.category}, {meta.value} pts)"
    )

    # ------------------------------------------------------------------
    # CTFd
    # ------------------------------------------------------------------

    ctfd = CTFdClient(
        base_url=settings.ctfd_url,
        token=settings.ctfd_token,
        username=settings.ctfd_user,
        password=settings.ctfd_pass,
    )

    cost_tracker = CostTracker()

    # ------------------------------------------------------------------
    # Create ONE solver
    # ------------------------------------------------------------------

    solver = Solver(
        model_spec=model_spec,
        challenge_dir=str(challenge_path),
        meta=meta,
        ctfd=ctfd,
        cost_tracker=cost_tracker,
        settings=settings,
    )

    # ------------------------------------------------------------------
    # Configure solver dependencies
    # ------------------------------------------------------------------

    solver.deps.no_submit = no_submit

    solver.deps.model_spec = model_spec

    async def submit_flag(flag: str):
        """Submit a flag through CTFd."""

        if no_submit:
            return "Dry-run mode: flag submission disabled.", False

        result = await ctfd.submit_flag(
            meta.name,
            flag,
        )

        return (
            result.display,
            result.status in ("correct", "already_solved"),
        )

    solver.deps.submit_fn = submit_flag

    # ------------------------------------------------------------------
    # RUN
    #
    # This is the actual performing process:
    #
    # Solver.start()
    #      ↓
    # PydanticAI
    #      ↓
    # LLM
    #      ↓
    # tools
    #      ↓
    # investigation
    #      ↓
    # flag
    # ------------------------------------------------------------------

    try:
        console.print(
            "\n[bold cyan]"
            "Starting ONE solver..."
            "[/bold cyan]"
        )

        console.print(
            "[dim]"
            "The solver will autonomously inspect, "
            "execute tools, investigate, and attempt "
            "to obtain the flag."
            "[/dim]\n"
        )

        result = await solver.run_until_done_or_gave_up()

        # --------------------------------------------------------------
        # Result
        # --------------------------------------------------------------

        if (
            result
            and result.status == FLAG_FOUND
        ):
            console.print(
                "\n[bold green]"
                "FLAG FOUND:"
                "[/bold green] "
                f"{result.flag}"
            )

        else:
            console.print(
                "\n[bold red]"
                "No confirmed flag found."
                "[/bold red]"
            )

            if result and result.flag:
                console.print(
                    f"[yellow]Candidate:[/yellow] "
                    f"{result.flag}"
                )

        # --------------------------------------------------------------
        # Solver information
        # --------------------------------------------------------------

        if result:
            console.print(
                "\n[bold]Solver Result[/bold]"
            )

            console.print(
                f"  Status: {result.status}"
            )

            console.print(
                f"  Steps: {result.step_count}"
            )

            console.print(
                f"  Cost: ${result.cost_usd:.4f}"
            )

            if result.findings_summary:
                console.print(
                    "\n[bold]Findings:[/bold]"
                )
                console.print(
                    result.findings_summary
                )

            if result.log_path:
                console.print(
                    "\n[bold]Trace:[/bold]"
                )
                console.print(
                    f"  {result.log_path}"
                )

        # --------------------------------------------------------------
        # Cost
        # --------------------------------------------------------------

        console.print(
            "\n[bold]Cost Summary:[/bold]"
        )

        for agent_name in cost_tracker.by_agent:
            console.print(
                "  "
                f"{agent_name}: "
                f"{cost_tracker.format_usage(agent_name)}"
            )

        console.print(
            "  [bold]"
            f"Total: ${cost_tracker.total_cost_usd:.4f}"
            "[/bold]"
        )

    finally:
        await solver.stop()
        await ctfd.close()


# ============================================================================
# INTERACTIVE SINGLE CHALLENGE
# ============================================================================


async def _run_interactive(
    settings: Settings,
    model_spec: str,
    no_submit: bool,
    description: str,
    hint: str,
    evidence: list[str],
) -> None:
    """Run exactly one interactive CTF challenge with one Solver."""

    from backend.agents.solver import Solver
    from backend.cost_tracker import CostTracker
    from backend.ctfd import CTFdClient
    from backend.prompts import ChallengeMeta
    from backend.sandbox import (
        cleanup_orphan_containers,
        configure_semaphore,
    )
    from backend.solver_base import FLAG_FOUND

    # ------------------------------------------------------------------
    # Display challenge
    # ------------------------------------------------------------------

    console.print(
        Panel(
            "[bold green]Challenge received.[/bold green]\n\n"
            f"[bold]Description:[/bold]\n"
            f"{description}\n\n"
            f"[bold]Hint:[/bold]\n"
            f"{hint or '(none)'}\n\n"
            f"[bold]Evidence:[/bold]\n"
            + (
                "\n".join(
                    f"• {path}"
                    for path in evidence
                )
                if evidence
                else "(none)"
            ),
            title="ONE-AGENT SOLVER",
            border_style="green",
        )
    )

    # ------------------------------------------------------------------
    # Make operator information available to Solver
    # ------------------------------------------------------------------

    _configure_solver_context(
        settings=settings,
        description=description,
        hint=hint,
        evidence=evidence,
    )

    # ------------------------------------------------------------------
    # Build interactive metadata
    # ------------------------------------------------------------------

    meta = ChallengeMeta(
        name="interactive",
        category="web",
        value=0,
        description=description.strip(),
        tags=["interactive"],
        connection_info="",
        hints=(
            [{"content": hint.strip()}]
            if hint.strip()
            else []
        ),
        solves=0,
    )

    # ------------------------------------------------------------------
    # ONE sandbox
    # ------------------------------------------------------------------

    configure_semaphore(1)

    await cleanup_orphan_containers()

    challenge_dir = str(
        Path.cwd()
    )

    # ------------------------------------------------------------------
    # CTFd
    # ------------------------------------------------------------------

    ctfd = CTFdClient(
        base_url=settings.ctfd_url,
        token=settings.ctfd_token,
        username=settings.ctfd_user,
        password=settings.ctfd_pass,
    )

    cost_tracker = CostTracker()

    # ------------------------------------------------------------------
    # ONE Solver
    # ------------------------------------------------------------------

    solver = Solver(
        model_spec=model_spec,
        challenge_dir=challenge_dir,
        meta=meta,
        ctfd=ctfd,
        cost_tracker=cost_tracker,
        settings=settings,
    )

    solver.deps.no_submit = no_submit
    solver.deps.model_spec = model_spec

    async def submit_flag(flag: str):
        """Submit a flag through CTFd."""

        if no_submit:
            return "Dry-run mode: flag submission disabled.", False

        result = await ctfd.submit_flag(
            meta.name,
            flag,
        )

        return (
            result.display,
            result.status in ("correct", "already_solved"),
        )

    solver.deps.submit_fn = submit_flag

    # ------------------------------------------------------------------
    # RUN ONE AGENT
    # ------------------------------------------------------------------

    try:
        console.print(
            "\n[bold cyan]"
            "Starting ONE autonomous solver..."
            "[/bold cyan]\n"
        )

        result = await solver.run_until_done_or_gave_up()

        # ------------------------------------------------------------------
        # Result
        # ------------------------------------------------------------------

        if (
            result
            and result.status == FLAG_FOUND
        ):
            console.print(
                "\n[bold green]"
                "FLAG FOUND:"
                "[/bold green] "
                f"{result.flag}"
            )

        else:
            console.print(
                "\n[bold red]"
                "No confirmed flag found."
                "[/bold red]"
            )

            if result and result.flag:
                console.print(
                    f"[yellow]Candidate:[/yellow] "
                    f"{result.flag}"
                )

        # ------------------------------------------------------------------
        # Details
        # ------------------------------------------------------------------

        if result:
            console.print(
                "\n[bold]Solver Result[/bold]"
            )

            console.print(
                f"  Status: {result.status}"
            )

            console.print(
                f"  Steps: {result.step_count}"
            )

            console.print(
                f"  Cost: ${result.cost_usd:.4f}"
            )

            if result.findings_summary:
                console.print(
                    "\n[bold]Findings:[/bold]"
                )
                console.print(
                    result.findings_summary
                )

            if result.log_path:
                console.print(
                    "\n[bold]Trace:[/bold]"
                )
                console.print(
                    f"  {result.log_path}"
                )

        # ------------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------------

        console.print(
            "\n[bold]Cost Summary:[/bold]"
        )

        for agent_name in cost_tracker.by_agent:
            console.print(
                "  "
                f"{agent_name}: "
                f"{cost_tracker.format_usage(agent_name)}"
            )

        console.print(
            "  [bold]"
            f"Total: ${cost_tracker.total_cost_usd:.4f}"
            "[/bold]"
        )

    finally:
        await solver.stop()
        await ctfd.close()


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()
