"""Pydantic AI tool wrappers for sandbox operations."""

from __future__ import annotations

from pydantic_ai import RunContext

from backend.deps import SolverDeps
from backend.tools.core import (
    do_bash,
    do_check_findings,
    do_list_files,
    do_read_file,
    do_web_fetch,
    do_webhook_create,
    do_webhook_get_requests,
    do_write_file,
)


async def bash(
    ctx: RunContext[SolverDeps],
    command: str,
    timeout_seconds: int = 60,
) -> str:
    """Execute a bash command inside the sandbox.

    Distfiles are read-only.
    Use /challenge/workspace/ for generated files.
    """
    return await do_bash(
        ctx.deps.sandbox,
        command,
        timeout_seconds,
    )


async def read_file(
    ctx: RunContext[SolverDeps],
    path: str,
) -> str:
    """Read a file from the challenge container."""
    return await do_read_file(
        ctx.deps.sandbox,
        path,
    )


async def write_file(
    ctx: RunContext[SolverDeps],
    path: str,
    content: str,
) -> str:
    """Write a file into the writable workspace."""
    return await do_write_file(
        ctx.deps.sandbox,
        path,
        content,
    )


async def list_files(
    ctx: RunContext[SolverDeps],
    path: str = "/challenge/distfiles",
) -> str:
    """List files in a challenge directory."""
    return await do_list_files(
        ctx.deps.sandbox,
        path,
    )


async def inspect_distfiles(
    ctx: RunContext[SolverDeps],
) -> str:
    """Perform a fast first-pass inspection of challenge files.

    This is the preferred first action for file-based challenges.

    It:
    - lists all files, including hidden files
    - identifies file types
    - prints small text files
    - searches for flag-like strings
    - avoids recursively scanning the entire filesystem
    - keeps output compact
    """

    command = r"""
set -o pipefail

DIR="/challenge/distfiles"

echo "=== FILES ==="
ls -la "$DIR"

echo
echo "=== FILE TYPES ==="
find "$DIR" -maxdepth 1 -type f -print0 2>/dev/null |
while IFS= read -r -d '' f; do
    printf '%s: ' "$f"
    file -b "$f" 2>/dev/null
done

echo
echo "=== SMALL TEXT FILES ==="

find "$DIR" -maxdepth 1 -type f -size -64k -print0 2>/dev/null |
while IFS= read -r -d '' f; do
    if file -b "$f" 2>/dev/null | grep -Eiq \
        'text|ascii|utf-8|json|xml|source|script|empty'
    then
        name="$(basename "$f")"

        echo "----- $name -----"

        sed -n '1,160p' "$f" 2>/dev/null
        echo
    fi
done

echo "=== FLAG SEARCH ==="

grep -RInaE \
    --exclude-dir=.git \
    --exclude='*.pyc' \
    '[A-Za-z0-9_]{2,32}\{[^{}[:cntrl:]]{1,256}\}' \
    "$DIR" 2>/dev/null |
head -50
"""

    return await do_bash(
        ctx.deps.sandbox,
        command,
        timeout_seconds=30,
    )


async def check_findings(
    ctx: RunContext[SolverDeps],
) -> str:
    """Check findings from sibling agents."""
    return await do_check_findings(
        ctx.deps.message_bus,
        ctx.deps.model_spec,
    )


async def notify_coordinator(
    ctx: RunContext[SolverDeps],
    message: str,
) -> str:
    """Send an important discovery to the coordinator."""

    if not ctx.deps.notify_coordinator:
        return "No coordinator connected."

    try:
        await ctx.deps.notify_coordinator(message)
        return "Message sent to coordinator."

    except Exception as exc:
        return f"Notification failed: {exc}"


async def web_fetch(
    ctx: RunContext[SolverDeps],
    url: str,
    method: str = "GET",
    body: str = "",
) -> str:
    """Fetch a URL from the host.

    For authenticated web interaction, prefer bash+curl
    inside the sandbox.
    """

    return await do_web_fetch(
        url,
        method,
        body,
    )


async def webhook_create(
    ctx: RunContext[SolverDeps],
) -> str:
    """Create a webhook.site token."""

    return await do_webhook_create()


async def webhook_get_requests(
    ctx: RunContext[SolverDeps],
    uuid: str,
) -> str:
    """Retrieve requests received by a webhook.site token."""

    return await do_webhook_get_requests(uuid)
