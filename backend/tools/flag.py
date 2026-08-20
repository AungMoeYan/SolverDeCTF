"""Flag submission tool."""

from pydantic_ai import RunContext

from backend.deps import SolverDeps
from backend.tools.core import do_submit_flag


async def submit_flag(
    ctx: RunContext[SolverDeps],
    flag: str,
) -> str:
    """Submit a flag to CTFd and mark confirmed candidates."""

    flag = flag.strip()

    if not flag:
        return "Empty flag — nothing to submit."

    # -----------------------------------------------------------
    # Dry-run mode
    # -----------------------------------------------------------

    if ctx.deps.no_submit:
        ctx.deps.confirmed_flag = flag

        return (
            f'DRY RUN — candidate flag "{flag}" accepted for testing. '
            "FLAG CONFIRMED. Stop solving."
        )

    # -----------------------------------------------------------
    # Actual submission
    # -----------------------------------------------------------

    if ctx.deps.submit_fn:
        display, is_confirmed = await ctx.deps.submit_fn(flag)
    else:
        display, is_confirmed = await do_submit_flag(
            ctx.deps.ctfd,
            ctx.deps.challenge_name,
            flag,
        )

    if is_confirmed:
        ctx.deps.confirmed_flag = flag

        return (
            f"{display}\n"
            "FLAG CONFIRMED. Stop solving."
        )

    return display
