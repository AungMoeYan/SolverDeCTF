"""CTF Agent CLI entry point.

Supports two modes:

1. Interactive mode:
       ctf-solve

   The operator provides:
       - challenge description
       - optional hint
       - optional evidence paths

2. Existing challenge-directory mode:
       ctf-solve --challenge test-challenges/recon-test

The interactive mode is intended to become the normal operator interface,
while the existing command-line options remain available for compatibility.
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


def _interactive_session() -> tuple[
    str,
    str,
    list[str],
]:
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
    help="CTFd URL (overrides .env)",
)
@click.option(
    "--ctfd-token",
    default=None,
    help="CTFd API token (overrides .env)",
)
@click.option(
    "--image",
    default="ctf-sandbox",
    help="Docker sandbox image name",
)
@click.option(
    "--models",
    multiple=True,
    help="Model specs (default: all configured)",
)
@click.option(
    "--challenge",
    default=None,
    help="Solve an existing challenge directory",
)
@click.option(
    "--challenges-dir",
    default="challenges",
    help="Directory for challenge files",
)
@click.option(
    "--instruction",
    default="",
    help="Extra instruction for the solver",
)
@click.option(
    "--description",
    default="",
    help="Challenge description",
)
@click.option(
    "--hint",
    default="",
    help="Challenge hint",
)
@click.option(
    "--evidence",
    multiple=True,
    help="Host evidence file/directory",
)
@click.option(
    "--no-submit",
    is_flag=True,
    help="Dry run — don't submit flags",
)
@click.option(
    "--coordinator-model",
    default=None,
    help="Model for coordinator",
)
@click.option(
    "--coordinator",
    default="claude",
    type=click.Choice(["claude", "codex"]),
    help="Coordinator backend",
)
@click.option(
    "--max-challenges",
    default=10,
    type=int,
    help="Max challenges solved concurrently",
)
@click.option(
    "--msg-port",
    default=0,
    type=int,
    help="Operator message port (0 = auto)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Verbose logging",
)
def main(
    ctfd_url: str | None,
    ctfd_token: str | None,
    image: str,
    models: tuple[str, ...],
    challenge: str | None,
    challenges_dir: str,
    instruction: str,
    description: str,
    hint: str,
    evidence: tuple[str, ...],
    no_submit: bool,
    coordinator_model: str | None,
    coordinator: str,
    max_challenges: int,
    msg_port: int,
    verbose: bool,
) -> None:
    """CTF Agent.

    With no --challenge argument, starts the interactive solver.

    Examples:

        ctf-solve

        ctf-solve --challenge test-challenges/recon-test

        ctf-solve --description "Find SSTI flag" --hint "Try Jinja2"

        ctf-solve --description "Analyze disk" \\
            --evidence /home/kali/Downloads/disk.dd
    """

    _setup_logging(verbose)

    settings = Settings(
        sandbox_image=image,
    )

    if ctfd_url:
        settings.ctfd_url = ctfd_url

    if ctfd_token:
        settings.ctfd_token = ctfd_token

    settings.max_concurrent_challenges = max_challenges

    # Keep the existing operator instruction mechanism.
    object.__setattr__(
        settings,
        "solver_instruction",
        instruction.strip(),
    )

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    model_specs = (
        list(models)
        if models
        else list(DEFAULT_MODELS)
    )

    # ------------------------------------------------------------------
    # Interactive mode
    #
    # If the operator simply runs:
    #
    #     ctf-solve
    #
    # collect the challenge information interactively.
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
    # Display startup information
    # ------------------------------------------------------------------

    console.print(
        "\n[bold]CTF Agent v2[/bold]"
    )

    console.print(
        f"  Models: {', '.join(model_specs)}"
    )

    console.print(
        f"  Image: {settings.sandbox_image}"
    )

    console.print(
        f"  Max challenges: {max_challenges}"
    )

    if description.strip():
        console.print(
            "  Description: "
            f"[cyan]{description.strip()[:500]}[/cyan]"
        )

    if hint.strip():
        console.print(
            "  Hint: "
            f"[cyan]{hint.strip()[:500]}[/cyan]"
        )

    if evidence:
        console.print(
            "  Evidence:"
        )

        for path in evidence:
            console.print(
                f"    [cyan]{path}[/cyan]"
            )

    console.print()

    # ------------------------------------------------------------------
    # Existing challenge-directory mode
    # ------------------------------------------------------------------

    if challenge:
        asyncio.run(
            _run_single(
                settings=settings,
                challenge_dir=challenge,
                model_specs=model_specs,
                no_submit=no_submit,
                max_challenges=max_challenges,
                description=description,
                hint=hint,
                evidence=list(evidence),
            )
        )

        return

    # ------------------------------------------------------------------
    # Interactive solver mode
    # ------------------------------------------------------------------

    asyncio.run(
        _run_interactive(
            settings=settings,
            model_specs=model_specs,
            no_submit=no_submit,
            max_challenges=max_challenges,
            description=description,
            hint=hint,
            evidence=list(evidence),
        )
    )


# ============================================================================
# EXISTING SINGLE CHALLENGE MODE
# ============================================================================


async def _run_single(
    settings: Settings,
    challenge_dir: str,
    model_specs: list[str],
    no_submit: bool,
    max_challenges: int,
    description: str = "",
    hint: str = "",
    evidence: list[str] | None = None,
) -> None:
    """Run an existing challenge with a solver swarm."""

    from backend.agents.swarm import ChallengeSwarm
    from backend.cost_tracker import CostTracker
    from backend.ctfd import CTFdClient
    from backend.prompts import ChallengeMeta
    from backend.sandbox import (
        cleanup_orphan_containers,
        configure_semaphore,
    )

    evidence = evidence or []

    max_containers = (
        max_challenges * len(model_specs)
    )

    configure_semaphore(
        max_containers
    )

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
        sys.exit(1)

    meta = ChallengeMeta.from_yaml(
        meta_path
    )

    console.print(
        f"[bold]Challenge:[/bold] "
        f"{meta.name} "
        f"({meta.category}, {meta.value} pts)"
    )

    ctfd = CTFdClient(
        base_url=settings.ctfd_url,
        token=settings.ctfd_token,
        username=settings.ctfd_user,
        password=settings.ctfd_pass,
    )

    cost_tracker = CostTracker()

    # ---------------------------------------------------------------
    # Preserve operator information.
    #
    # ChallengeSwarm will need to forward these values to Solver.
    # ---------------------------------------------------------------

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

    swarm = ChallengeSwarm(
        challenge_dir=str(
            challenge_path
        ),
        meta=meta,
        ctfd=ctfd,
        cost_tracker=cost_tracker,
        settings=settings,
        model_specs=model_specs,
        no_submit=no_submit,
    )

    try:
        result = await swarm.run()

        from backend.solver_base import FLAG_FOUND

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
                "No flag found."
                "[/bold red]"
            )

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
            f"Total: ${cost_tracker.total_cost_usd:.2f}"
            "[/bold]"
        )

    finally:
        await ctfd.close()


async def _run_interactive(
    settings: Settings,
    model_specs: list[str],
    no_submit: bool,
    max_challenges: int,
    description: str,
    hint: str,
    evidence: list[str],
) -> None:
    """Run the solver using operator-provided challenge information."""

    from backend.agents.swarm import ChallengeSwarm
    from backend.cost_tracker import CostTracker
    from backend.ctfd import CTFdClient
    from backend.prompts import ChallengeMeta
    from backend.sandbox import (
        cleanup_orphan_containers,
        configure_semaphore,
    )

    # ---------------------------------------------------------------
    # Sandbox concurrency
    # ---------------------------------------------------------------

    max_containers = (
        max_challenges * len(model_specs)
    )

    configure_semaphore(
        max_containers
    )

    await cleanup_orphan_containers()

    # ---------------------------------------------------------------
    # Display operator input
    # ---------------------------------------------------------------

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
            title="Solver Input",
            border_style="green",
        )
    )

    # ---------------------------------------------------------------
    # Build an in-memory ChallengeMeta
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Connect operator information to SolverDeps
    #
    # Solver currently reads the operator instruction from
    # settings.solver_instruction.
    #
    # Keep the complete interactive input together so the model
    # receives the actual operator context.
    # ---------------------------------------------------------------

    instruction_parts: list[str] = []

    if description.strip():
        instruction_parts.append(
            f"Operator challenge description:\n"
            f"{description.strip()}"
        )

    if hint.strip():
        instruction_parts.append(
            f"Operator hint:\n"
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
        "\n\n".join(
            instruction_parts
        ),
    )

    # Preserve these fields too for compatibility with the existing
    # single-challenge path and future evidence integration.
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

    # ---------------------------------------------------------------
    # CTFd
    # ---------------------------------------------------------------

    ctfd = CTFdClient(
        base_url=settings.ctfd_url,
        token=settings.ctfd_token,
        username=settings.ctfd_user,
        password=settings.ctfd_pass,
    )

    cost_tracker = CostTracker()

    # ---------------------------------------------------------------
    # Interactive challenge directory
    #
    # The solver expects a challenge_dir. For an interactive
    # challenge there may be no distfiles, so use the project root.
    #
    # DockerSandbox will still start normally.
    # ---------------------------------------------------------------

    challenge_dir = str(
        Path.cwd()
    )

    # ---------------------------------------------------------------
    # Swarm
    # ---------------------------------------------------------------

    swarm = ChallengeSwarm(
        challenge_dir=challenge_dir,
        meta=meta,
        ctfd=ctfd,
        cost_tracker=cost_tracker,
        settings=settings,
        model_specs=model_specs,
        no_submit=no_submit,
    )

    try:
        console.print(
            "\n[bold cyan]Starting interactive solver swarm...[/bold cyan]"
        )

        result = await swarm.run()

        from backend.solver_base import FLAG_FOUND

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
                "No flag found."
                "[/bold red]"
            )

    finally:
        await ctfd.close()

# ============================================================================
# COORDINATOR MESSAGE COMMAND
# ============================================================================


@click.command()
@click.argument("message")
@click.option(
    "--port",
    default=9400,
    type=int,
    help="Coordinator message port",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Coordinator host",
)
def msg(
    message: str,
    port: int,
    host: str,
) -> None:
    """Send a message to the running coordinator."""

    import json
    import urllib.request

    body = json.dumps(
        {"message": message}
    ).encode()

    req = urllib.request.Request(
        f"http://{host}:{port}/msg",
        data=body,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req,
            timeout=5,
        ) as resp:
            data = json.loads(
                resp.read()
            )

            console.print(
                "[green]Sent:[/green] "
                f"{data.get('queued', message[:200])}"
            )

    except Exception as exc:
        console.print(
            "[red]Failed:[/red] "
            f"{exc}"
        )

        console.print(
            "Is the coordinator running?"
        )

        sys.exit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()
