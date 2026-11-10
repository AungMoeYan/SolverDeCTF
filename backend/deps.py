
"""Shared dependency types — avoids circular imports between agents and tools."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient

if TYPE_CHECKING:
    from backend.investigation import InvestigationState
    from backend.message_bus import ChallengeMessageBus
    from backend.sandbox import DockerSandbox


# Type for the deduped submit callback:
# (flag) -> (display, is_confirmed)
SubmitFn = Callable[
    [str],
    Coroutine[
        Any,
        Any,
        tuple[str, bool],
    ],
]


@dataclass
class SolverDeps:
    """Runtime dependencies available to solver tools."""

    sandbox: DockerSandbox

    ctfd: CTFdClient

    challenge_dir: str

    challenge_name: str

    workspace_dir: str

    use_vision: bool

    cost_tracker: CostTracker | None = None

    confirmed_flag: str | None = None

    message_bus: ChallengeMessageBus | None = None

    model_spec: str = ""

    submit_fn: SubmitFn | None = None

    no_submit: bool = False

    notify_coordinator: (
        Callable[
            [str],
            Coroutine[Any, Any, None],
        ]
        | None
    ) = None

    # ---------------------------------------------------------------
    # Operator instruction
    # ---------------------------------------------------------------

    instruction: str = ""

    investigation: InvestigationState | None = None
