"""Focused tests for solver investigation orchestration."""

from __future__ import annotations

from backend.investigation import (
    ActionTracker,
    ChallengeCategory,
    InvestigationPhase,
    InvestigationState,
    STUCK_THRESHOLD,
    action_signature,
    apply_tool_result,
    classify_challenge,
    compact_summary,
    extract_flag_candidate,
    mark_confirmed_flag,
    should_inspect_distfiles,
    state_from_challenge,
)


UPLOAD_DESC = (
    "A developer has added profile picture upload functionality to a website. "
    "However, the implementation is flawed, and it presents an opportunity to you. "
    "Your mission, should you choose to accept it, is to navigate to the provided "
    "web page and locate the file upload area. Your ultimate goal is to find the "
    "hidden flag located in the /root directory.\n"
    "http://standard-pizzas.picoctf.net:54152/"
)


def test_classify_upload_url_as_web() -> None:
    category = classify_challenge(
        UPLOAD_DESC,
        challenge_name="interactive",
        metadata_category="web",
    )
    assert category is ChallengeCategory.WEB


def test_action_normalization_equivalent_commands() -> None:
    left = action_signature("bash", {"command": "cat /root/flag.txt"})
    right = action_signature("bash", {"command": "  cat   /root/flag.txt  "})
    assert left == right


def test_duplicate_prevention() -> None:
    state = InvestigationState()
    tracker = ActionTracker()
    args = {"command": "curl http://example.test/robots.txt"}
    signature = action_signature("bash", args)
    assert tracker.is_duplicate(state, signature) is False
    tracker.record(state, "bash", signature)
    assert tracker.is_duplicate(state, signature) is True


def test_whoami_updates_execution_user() -> None:
    state = InvestigationState()
    progress = apply_tool_result(
        state,
        "bash",
        {"command": "whoami"},
        "www-data\n",
    )
    assert progress is True
    assert state.privileges["execution_user"] == "www-data"
    assert state.execution_context["execution_user"] == "www-data"


def test_web_rce_then_denied_moves_to_privesc() -> None:
    state = InvestigationState(category=ChallengeCategory.WEB)
    apply_tool_result(
        state,
        "bash",
        {"command": "curl http://target/uploads/shell.php"},
        "uid=33(www-data) gid=33(www-data)\n",
    )
    assert state.execution_context.get("web_rce") is True
    assert state.privileges.get("execution_user") == "www-data"
    apply_tool_result(
        state,
        "bash",
        {"command": "cat /root/flag.txt"},
        "cat: /root/flag.txt: Permission denied\n",
    )
    assert state.phase is InvestigationPhase.PRIVILEGE_ESCALATION
    assert state.privileges.get("direct_root_access") is False


def test_no_progress_reaches_stuck() -> None:
    state = InvestigationState()
    for _ in range(STUCK_THRESHOLD):
        apply_tool_result(
            state,
            "bash",
            {"command": "true"},
            "(no output)",
        )
    assert state.consecutive_no_progress >= STUCK_THRESHOLD
    assert state.phase is InvestigationPhase.STUCK


def test_flag_detection() -> None:
    assert extract_flag_candidate("prefix picoCTF{example} suffix") == "picoCTF{example}"
    assert extract_flag_candidate("CTF{flag}") is None


def test_confirmed_flag_completes() -> None:
    state = InvestigationState()
    mark_confirmed_flag(state, "picoCTF{done}")
    assert state.confirmed_flag == "picoCTF{done}"
    assert state.phase is InvestigationPhase.COMPLETED


def test_compaction_preserves_state() -> None:
    state = state_from_challenge(UPLOAD_DESC, challenge_name="interactive")
    apply_tool_result(
        state,
        "web_fetch",
        {"url": "http://standard-pizzas.picoctf.net:54152/"},
        "HTTP 200 OK\nupload.php form",
    )
    apply_tool_result(
        state,
        "bash",
        {"command": "whoami"},
        "www-data\n",
    )
    state.retry_count = 1
    summary = compact_summary(state)
    assert "WEB" in summary.upper() or "web" in summary
    assert "www-data" in summary
    assert "standard-pizzas.picoctf.net" in summary
    assert "Do not repeat completed actions" in summary
    assert "PRIVILEGE" in summary or "POST_EXPLOITATION" in summary or "DISCOVERY" in summary


def test_web_challenge_does_not_auto_inspect_distfiles() -> None:
    state = state_from_challenge(UPLOAD_DESC, challenge_name="interactive")
    assert state.category is ChallengeCategory.WEB
    assert state.target_urls
    assert should_inspect_distfiles(state) is False
