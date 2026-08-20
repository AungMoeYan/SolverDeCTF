"""Tests for the new investigation state, classification, dedup, and flag detection logic.

These test the pure functions and InvestigationState class from backend.investigation
(no pydantic_ai/boto3 dependency) and the prompt integration from backend.prompts.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.investigation import (
    InvestigationState,
    build_state_context,
    classify_challenge,
    detect_flag_in_text,
    extract_target,
    is_readonly_action,
    normalize_action,
    normalize_url,
)
from backend.prompts import ChallengeMeta, build_prompt


# ===========================================================================
# Challenge Classification
# ===========================================================================


class TestClassifyChallenge:
    def test_explicit_category(self):
        meta = ChallengeMeta(category="web", name="Test")
        assert classify_challenge(meta, []) == "web"

    def test_explicit_category_reverse(self):
        meta = ChallengeMeta(category="reverse", name="Test")
        assert classify_challenge(meta, []) == "reverse"

    def test_tag_classification(self):
        meta = ChallengeMeta(tags=["xss", "sqli"], name="Test")
        assert classify_challenge(meta, []) == "web"

    def test_tag_classification_crypto(self):
        meta = ChallengeMeta(tags=["RSA", "padding"], name="Test")
        assert classify_challenge(meta, []) == "crypto"

    def test_connection_info_http(self):
        meta = ChallengeMeta(connection_info="http://challenge.ctf.example.com:8080")
        assert classify_challenge(meta, []) == "web"

    def test_connection_info_nc(self):
        meta = ChallengeMeta(connection_info="nc challenge.ctf.example.com 9001")
        assert classify_challenge(meta, []) == "pwn"

    def test_connection_info_nc_crypto_hint(self):
        meta = ChallengeMeta(
            connection_info="nc challenge.ctf.example.com 9001",
            description="Decrypt the RSA-encrypted messages from the oracle",
        )
        assert classify_challenge(meta, []) == "crypto"

    def test_description_keywords(self):
        meta = ChallengeMeta(description="Upload a PHP file to gain RCE")
        assert classify_challenge(meta, []) == "web"

    def test_description_keywords_pwn(self):
        meta = ChallengeMeta(description="Exploit the buffer overflow in the binary")
        assert classify_challenge(meta, []) == "pwn"

    def test_distfile_extensions_web(self):
        meta = ChallengeMeta(name="Test")
        assert classify_challenge(meta, ["index.php", "config.html"]) == "web"

    def test_distfile_extensions_reverse(self):
        meta = ChallengeMeta(name="Test")
        assert classify_challenge(meta, ["binary.elf"]) == "reverse"

    def test_distfile_extensions_forensics(self):
        meta = ChallengeMeta(name="Test")
        assert classify_challenge(meta, ["disk.img"]) == "forensics"

    def test_distfile_extensions_stego(self):
        meta = ChallengeMeta(name="Test")
        assert classify_challenge(meta, ["image.png"]) == "steganography"

    def test_distfile_extensions_misc(self):
        meta = ChallengeMeta(name="Test")
        assert classify_challenge(meta, ["solve.py"]) == "misc"

    def test_fallback_misc(self):
        meta = ChallengeMeta(name="Test")
        assert classify_challenge(meta, []) == "misc"

    def test_priority_category_over_tags(self):
        """Explicit category should win over tags."""
        meta = ChallengeMeta(category="crypto", tags=["web", "upload"])
        assert classify_challenge(meta, []) == "crypto"

    def test_priority_category_over_connection(self):
        """Explicit category should win over connection_info heuristics."""
        meta = ChallengeMeta(category="pwn", connection_info="http://example.com")
        assert classify_challenge(meta, []) == "pwn"


# ===========================================================================
# Target Extraction
# ===========================================================================


class TestExtractTarget:
    def test_http_url(self):
        assert extract_target("http://example.com:8080") == "http://example.com:8080"

    def test_https_url(self):
        assert extract_target("https://example.com/path") == "https://example.com/path"

    def test_nc(self):
        result = extract_target("nc example.com 9001")
        assert "example.com" in result
        assert "9001" in result

    def test_empty(self):
        assert extract_target("") == ""

    def test_none_like(self):
        assert extract_target("   ") == ""


# ===========================================================================
# Flag Detection
# ===========================================================================


class TestDetectFlagInText:
    def test_standard_flag(self):
        text = "Congratulations! flag{abc123_xyz}"
        flags = detect_flag_in_text(text)
        assert len(flags) == 1
        assert "flag{abc123_xyz}" in flags

    def test_picoCTF(self):
        text = "picoCTF{s0me_fl4g}"
        flags = detect_flag_in_text(text)
        assert len(flags) == 1

    def test_CTF_uppercase(self):
        text = "CTF{An0th3r_Fl4g}"
        flags = detect_flag_in_text(text)
        assert len(flags) == 1

    def test_flag_with_equals(self):
        text = "flag=Th1s_1s_A_Fl4g_2024"
        flags = detect_flag_in_text(text)
        assert len(flags) >= 1

    def test_flag_with_colon(self):
        text = "flag: s0mE_fL4g_v4lue"
        flags = detect_flag_in_text(text)
        assert len(flags) >= 1

    def test_no_false_positive_placeholder(self):
        text = "the flag{placeholder} is not real"
        flags = detect_flag_in_text(text)
        assert "flag{placeholder}" not in flags

    def test_no_false_positive_flag_word(self):
        text = "this is a flag for the challenge"
        flags = detect_flag_in_text(text)
        assert len(flags) == 0

    def test_multiple_flags(self):
        text = "flag{first} and CTF{second}"
        flags = detect_flag_in_text(text)
        assert len(flags) == 2

    def test_empty_text(self):
        assert detect_flag_in_text("") == []

    def test_too_short(self):
        # flag{} has no content between braces — not a match.
        # flag{x} (7 chars) IS the shortest valid brace-format flag.
        text = "flag{}"
        flags = detect_flag_in_text(text)
        assert len(flags) == 0

    def test_shortest_valid_brace_flag(self):
        text = "flag{x}"
        flags = detect_flag_in_text(text)
        assert len(flags) == 1
        assert flags[0] == "flag{x}"

    def test_flag_in_html(self):
        text = '<div class="result">flag{h1dd3n_1n_htm1}</div>'
        flags = detect_flag_in_text(text)
        assert any("flag{h1dd3n_1n_htm1}" in f for f in flags)

    def test_flag_in_curl_output(self):
        text = "HTTP 200 OK\n<html><body>flag{curl_found_it}</body></html>"
        flags = detect_flag_in_text(text)
        assert len(flags) == 1


# ===========================================================================
# Action Normalization
# ===========================================================================


class TestNormalizeAction:
    def test_curl_get(self):
        norm = normalize_action("bash", {"command": "curl http://target.com/"})
        assert "curl:GET:http://target.com" in norm

    def test_curl_get_silent(self):
        n1 = normalize_action("bash", {"command": "curl -s http://target.com/"})
        n2 = normalize_action("bash", {"command": "curl http://target.com/"})
        assert n1 == n2

    def test_curl_post(self):
        norm = normalize_action("bash", {"command": "curl -d 'x=1' http://target.com/api"})
        assert "curl:POST" in norm

    def test_curl_post_data_raw(self):
        norm = normalize_action("bash", {"command": "curl --data-raw @payload http://target.com"})
        assert "curl:POST" in norm

    def test_file_inspect(self):
        norm = normalize_action("bash", {"command": "file /challenge/distfiles/binary"})
        assert norm == "inspect:file:/challenge/distfiles/binary"

    def test_strings_inspect(self):
        norm = normalize_action("bash", {"command": "strings -n 8 /challenge/distfiles/binary"})
        assert norm == "inspect:strings:/challenge/distfiles/binary"

    def test_ls(self):
        norm = normalize_action("bash", {"command": "ls -la /challenge/distfiles"})
        assert "list:/challenge/distfiles" in norm

    def test_cat(self):
        norm = normalize_action("bash", {"command": "cat /challenge/distfiles/readme.txt"})
        assert norm == "read:/challenge/distfiles/readme.txt"

    def test_web_fetch_get(self):
        norm = normalize_action("web_fetch", {"url": "http://target.com/page", "method": "GET"})
        assert "web_fetch:GET:http://target.com/page" in norm

    def test_list_files(self):
        norm = normalize_action("list_files", {"path": "/challenge/distfiles"})
        assert norm == "list_files:/challenge/distfiles"

    def test_read_file(self):
        norm = normalize_action("read_file", {"path": "/challenge/distfiles/f.txt"})
        assert norm == "read_file:/challenge/distfiles/f.txt"

    def test_write_file_normalized(self):
        norm = normalize_action("write_file", {"path": "/challenge/workspace/shell.php", "content": "<?php"})
        assert "shell.php" in norm

    def test_submit_flag_not_normalized_to_content(self):
        norm = normalize_action("submit_flag", {"flag": "flag{test}"})
        assert "flag{test}" in norm


# ===========================================================================
# Read-Only Detection
# ===========================================================================


class TestIsReadonlyAction:
    def test_bash_cat_is_readonly(self):
        assert is_readonly_action("bash", {"command": "cat /tmp/file"}) is True

    def test_bash_curl_get_is_readonly(self):
        assert is_readonly_action("bash", {"command": "curl http://target/"}) is True

    def test_bash_curl_post_is_not_readonly(self):
        assert is_readonly_action("bash", {"command": "curl -d 'x=1' http://target/"}) is False

    def test_bash_python_is_not_readonly(self):
        assert is_readonly_action("bash", {"command": "python3 exploit.py"}) is False

    def test_web_fetch_get_is_readonly(self):
        assert is_readonly_action("web_fetch", {"url": "http://target/", "method": "GET"}) is True

    def test_web_fetch_post_is_not_readonly(self):
        assert is_readonly_action("web_fetch", {"url": "http://target/", "method": "POST"}) is False

    def test_write_file_is_not_readonly(self):
        assert is_readonly_action("write_file", {"path": "/tmp/x", "content": "y"}) is False

    def test_list_files_is_readonly(self):
        assert is_readonly_action("list_files", {"path": "/challenge/distfiles"}) is True

    def test_submit_flag_is_not_readonly(self):
        assert is_readonly_action("submit_flag", {"flag": "test"}) is False

    def test_bash_file_is_readonly(self):
        assert is_readonly_action("bash", {"command": "file /challenge/distfiles/binary"}) is True

    def test_bash_nmap_is_readonly(self):
        assert is_readonly_action("bash", {"command": "nmap -sV target"}) is True


# ===========================================================================
# URL Normalization
# ===========================================================================


class TestNormalizeUrl:
    def test_strip_trailing_slash(self):
        assert normalize_url("http://target.com/") == "http://target.com"

    def test_strip_query(self):
        assert normalize_url("http://target.com/page?x=1&y=2") == "http://target.com/page"

    def test_strip_whitespace(self):
        assert normalize_url("  http://target.com  ") == "http://target.com"


# ===========================================================================
# Investigation State
# ===========================================================================


class TestInvestigationState:
    def test_add_discovery_new(self):
        state = InvestigationState()
        assert state.add_discovery("Upload works") is True
        assert len(state.discoveries) == 1

    def test_add_discovery_duplicate(self):
        state = InvestigationState()
        state.add_discovery("Upload works")
        assert state.add_discovery("Upload works") is False
        assert len(state.discoveries) == 1

    def test_add_discovery_truncates(self):
        state = InvestigationState()
        long_fact = "A" * 300
        state.add_discovery(long_fact)
        assert len(state.discoveries[0]) <= 200

    def test_add_discovery_resets_streak(self):
        state = InvestigationState()
        state._no_progress_streak = 5
        state.add_discovery("New info")
        assert state.no_progress_streak == 0

    def test_add_endpoint_new(self):
        state = InvestigationState()
        assert state.add_endpoint("http://target.com") is True

    def test_add_endpoint_duplicate(self):
        state = InvestigationState()
        state.add_endpoint("http://target.com")
        assert state.add_endpoint("http://target.com") is False

    def test_add_endpoint_normalizes(self):
        state = InvestigationState()
        state.add_endpoint("http://target.com/")
        assert state.add_endpoint("http://target.com") is False

    def test_record_and_has_action(self):
        state = InvestigationState()
        state.record_action("curl:GET:http://x", "HTTP 200 OK")
        found, prior = state.has_action("curl:GET:http://x")
        assert found is True
        assert "200" in prior

    def test_has_action_missing(self):
        state = InvestigationState()
        found, prior = state.has_action("nonexistent")
        assert found is False
        assert prior == ""

    def test_increment_no_progress(self):
        state = InvestigationState()
        assert state.increment_no_progress() == 1
        assert state.increment_no_progress() == 2
        assert state.no_progress_streak == 2

    def test_cache_eviction(self):
        state = InvestigationState()
        state._max_cache = 3
        for i in range(5):
            state.record_action(f"action_{i}", f"result_{i}")
        # Oldest entries evicted
        found, _ = state.has_action("action_0")
        assert found is False
        found, _ = state.has_action("action_4")
        assert found is True

    def test_flag_candidates(self):
        state = InvestigationState()
        state.flag_candidates.append("flag{first}")
        state.flag_candidates.append("flag{second}")
        assert len(state.flag_candidates) == 2


# ===========================================================================
# State Context Rendering
# ===========================================================================


class TestBuildStateContext:
    def test_basic_render(self):
        state = InvestigationState(category="web", target="http://target.com")
        ctx = build_state_context(state, 20)
        assert "web" in ctx
        assert "http://target.com" in ctx
        assert "20 requests remaining" in ctx

    def test_discoveries_rendered(self):
        state = InvestigationState()
        state.add_discovery("Upload works")
        state.add_discovery("PHP executes")
        ctx = build_state_context(state, 10)
        assert "Upload works" in ctx
        assert "PHP executes" in ctx

    def test_no_progress_warning(self):
        state = InvestigationState()
        state._no_progress_streak = 5
        ctx = build_state_context(state, 10)
        assert "NO PROGRESS" in ctx

    def test_budget_remaining(self):
        state = InvestigationState()
        ctx = build_state_context(state, 3)
        assert "3 requests remaining" in ctx

    def test_flag_candidates_rendered(self):
        state = InvestigationState()
        state.flag_candidates.append("flag{test}")
        ctx = build_state_context(state, 10)
        assert "flag{test}" in ctx

    def test_empty_state(self):
        state = InvestigationState()
        ctx = build_state_context(state, 25)
        assert "Investigation State" in ctx
        assert "Budget" in ctx

    def test_endpoints_truncated(self):
        state = InvestigationState()
        for i in range(15):
            state.add_endpoint(f"http://target.com/page{i}")
        ctx = build_state_context(state, 10)
        # Should show at most 8
        assert ctx.count("http://target.com/") <= 8


# ===========================================================================
# Prompt Integration
# ===========================================================================


class TestPromptIntegration:
    def test_category_hint_web(self):
        meta = ChallengeMeta(name="Test", category="web")
        prompt = build_prompt(meta, [], category_hint="web")
        assert "Category: Web" in prompt
        assert "PRIMARY target is the web service" in prompt

    def test_category_hint_crypto(self):
        meta = ChallengeMeta(name="Test", category="crypto")
        prompt = build_prompt(meta, [], category_hint="crypto")
        assert "Category: Crypto" in prompt

    def test_no_category_hint(self):
        meta = ChallengeMeta(name="Test", category="web")
        prompt = build_prompt(meta, [])
        assert "Category: Web" not in prompt

    def test_existing_prompt_unchanged_without_hint(self):
        """Without category_hint, prompt should be identical to before."""
        meta = ChallengeMeta(
            name="TestChallenge",
            category="web",
            value=100,
            description="Solve this challenge.",
            connection_info="http://host.docker.internal:8080",
        )
        prompt_old = build_prompt(meta, ["file.txt"])
        prompt_new = build_prompt(meta, ["file.txt"], category_hint="")
        # Empty string hint should produce same result as no hint
        assert prompt_old == prompt_new

    def test_category_hint_injected_before_instructions(self):
        meta = ChallengeMeta(name="Test", category="reverse")
        prompt = build_prompt(meta, ["binary.elf"], category_hint="reverse")
        # Category guidance should appear before Instructions
        cat_pos = prompt.index("Category: Reverse")
        inst_pos = prompt.index("## Instructions")
        assert cat_pos < inst_pos

    def test_binary_analysis_still_shown_for_reverse(self):
        meta = ChallengeMeta(name="Test", category="reverse")
        prompt = build_prompt(meta, ["binary.elf"], category_hint="reverse")
        assert "pyghidra" in prompt

    def test_binary_analysis_not_shown_for_web(self):
        meta = ChallengeMeta(name="Test", category="web")
        prompt = build_prompt(meta, [], category_hint="web")
        assert "pyghidra" not in prompt


# ===========================================================================
# F5 Fix: Flag pattern 2 tightened (min 12 chars)
# ===========================================================================


class TestFlagPattern2Tightened:
    """After F5, pattern 2 (flag: =) requires 12+ chars to avoid false positives."""

    def test_short_flag_colon_no_match(self):
        text = "Use the flag: check the source code"
        flags = detect_flag_in_text(text)
        assert len(flags) == 0

    def test_short_flag_equals_no_match(self):
        text = "flag=short"
        flags = detect_flag_in_text(text)
        assert len(flags) == 0

    def test_long_flag_colon_still_matches(self):
        text = "flag: s0mE_fL4g_v4lue"
        flags = detect_flag_in_text(text)
        assert len(flags) >= 1

    def test_long_flag_equals_still_matches(self):
        text = "flag=Th1s_1s_A_Fl4g_2024"
        flags = detect_flag_in_text(text)
        assert len(flags) >= 1

    def test_flag_word_in_sentence_no_match(self):
        text = "the flag for this challenge is hidden"
        flags = detect_flag_in_text(text)
        assert len(flags) == 0


# ===========================================================================
# F3 Fix: Cache invalidation after state-changing actions
# ===========================================================================


class TestCacheInvalidation:
    """After F3, the dedup cache is cleared on write_file and non-readonly bash."""

    def test_write_file_clears_cache(self):
        state = InvestigationState()
        state.record_action("read:/tmp/x", "old content")
        assert state.has_action("read:/tmp/x")[0] is True
        # Simulate what solver._update_state does
        state._action_cache.clear()
        assert state.has_action("read:/tmp/x")[0] is False

    def test_nonreadonly_bash_clears_cache(self):
        state = InvestigationState()
        state.record_action("curl:GET:http://target.com", "old page")
        state.record_action("list:/challenge/distfiles", "a.txt b.txt")
        assert len(state._action_cache) == 2
        # Simulate what solver._update_state does for python3 exploit.py
        state._action_cache.clear()
        assert len(state._action_cache) == 0

    def test_readonly_bash_does_not_clear_cache(self):
        state = InvestigationState()
        state.record_action("inspect:file:/tmp/binary", "ELF 64-bit")
        state.record_action("curl:GET:http://target.com", "HTTP 200")
        # Simulate a read-only action — cache should NOT be cleared
        # (the solver only clears for non-readonly actions)
        # We test this by just checking the cache is still populated
        assert len(state._action_cache) == 2

    def test_cache_refill_after_clear(self):
        """After clearing, new actions can be recorded and dedup'd again."""
        state = InvestigationState()
        state.record_action("read:/tmp/x", "old")
        state._action_cache.clear()
        assert state.has_action("read:/tmp/x")[0] is False
        # Re-execute the same read — should not be dedup'd
        state.record_action("read:/tmp/x", "new content after write")
        assert state.has_action("read:/tmp/x")[0] is True


# ===========================================================================
# F2 Fix: Confirmed flag preservation logic
# ===========================================================================


class TestConfirmedFlagPreservation:
    """Test the priority logic for flag selection (confirmed > candidate > none).

    This tests the state-level logic. The solver-level integration (checking
    deps.confirmed_flag) requires a full Solver mock and is tested implicitly
    via the TracingToolset flag detection flow.
    """

    def test_flag_candidate_takes_best(self):
        """If multiple candidates exist, the last one is used (most recent)."""
        state = InvestigationState()
        state.flag_candidates.append("flag{old_candidate}")
        state.flag_candidates.append("flag{new_candidate}")
        best = state.flag_candidates[-1]
        assert best == "flag{new_candidate}"

    def test_no_duplicate_candidates(self):
        """Flag detection in TracingToolset checks 'not in' before appending."""
        state = InvestigationState()
        state.flag_candidates.append("flag{abc}")
        # Simulating the check done in solver.py
        if "flag{abc}" not in state.flag_candidates:
            state.flag_candidates.append("flag{abc}")
        assert len(state.flag_candidates) == 1

    def test_empty_candidates_graceful(self):
        """Accessing candidates when empty should not error."""
        state = InvestigationState()
        assert len(state.flag_candidates) == 0
        # -1 index on empty list would raise, but solver checks len first
        assert not state.flag_candidates


# ===========================================================================
# F4 Fix: Budget exhaustion returns GAVE_UP, not ERROR
# ===========================================================================


class TestBudgetConstants:
    """Verify budget-related constants are sensible."""

    def test_default_budget_is_reasonable(self):
        from backend.investigation import DEFAULT_REQUEST_BUDGET
        assert isinstance(DEFAULT_REQUEST_BUDGET, int)
        assert 10 <= DEFAULT_REQUEST_BUDGET <= 100

    def test_state_context_shows_budget(self):
        state = InvestigationState()
        ctx = build_state_context(state, 0)
        assert "0 requests remaining" in ctx

    def test_state_context_negative_budget_shows_raw(self):
        state = InvestigationState()
        ctx = build_state_context(state, -5)
        # build_state_context renders the raw number; callers are responsible for clamping
        assert "-5 requests remaining" in ctx


# ===========================================================================
# Integration: TracingToolset dedup and cache invalidation
# ===========================================================================


@pytest.fixture
def mock_ctx():
    """Build a lightweight mock RunContext[SolverDeps] for TracingToolset tests."""
    from unittest.mock import MagicMock
    from pydantic_ai import RunContext
    from backend.deps import SolverDeps

    sandbox = MagicMock()
    sandbox.workspace_dir = "/workspace"
    ctfd = MagicMock()
    deps = SolverDeps(
        sandbox=sandbox,
        ctfd=ctfd,
        challenge_dir="/challenge/test",
        challenge_name="test_challenge",
        workspace_dir="/workspace",
        use_vision=False,
    )
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx


@pytest.fixture
def tracing_toolset(mock_ctx):
    """Build a TracingToolset with a thin mock wrapper for integration tests."""
    from unittest.mock import AsyncMock, MagicMock
    from backend.agents.solver import TracingToolset
    from backend.investigation import InvestigationState
    from backend.loop_detect import LoopDetector
    from backend.tracing import SolverTracer

    # The wrapped toolset just echoes back a fixed result
    mock_wrapped = MagicMock()
    mock_wrapped.call_tool = AsyncMock(return_value="mock result")

    tracer = MagicMock(spec=SolverTracer)
    tracer.tool_call = MagicMock()
    tracer.tool_result = MagicMock()
    tracer.event = MagicMock()

    loop_detector = LoopDetector()
    state = InvestigationState()

    toolset = TracingToolset(
        wrapped=mock_wrapped,
        tracer=tracer,
        loop_detector=loop_detector,
        step_counter=[0],
        state=state,
        request_budget=25,
    )
    return toolset


@pytest.fixture
def mock_tool():
    """A mock ToolsetTool."""
    from unittest.mock import MagicMock
    tool = MagicMock()
    tool.name = "test_tool"
    return tool


class TestTracingToolsetDedup:
    """Integration tests for dedup logic through TracingToolset.call_tool."""

    @pytest.mark.asyncio
    async def test_duplicate_readonly_actions_are_deduplicated(
        self, tracing_toolset, mock_ctx, mock_tool
    ):
        """Two equivalent read-only bash actions should be deduplicated on the second call."""
        args = {"command": "cat /challenge/distfiles/flag.txt"}

        # First call — should execute normally
        r1 = await tracing_toolset.call_tool("bash", args, mock_ctx, mock_tool)
        assert "mock result" in r1

        # Second identical call — should be dedup'd
        r2 = await tracing_toolset.call_tool("bash", args, mock_ctx, mock_tool)
        assert "[DEDUP]" in r2
        assert "Prior result" in r2

    @pytest.mark.asyncio
    async def test_different_readonly_actions_not_deduplicated(
        self, tracing_toolset, mock_ctx, mock_tool
    ):
        """Different read-only actions should NOT be deduplicated."""
        args1 = {"command": "cat /tmp/file_a.txt"}
        args2 = {"command": "cat /tmp/file_b.txt"}

        r1 = await tracing_toolset.call_tool("bash", args1, mock_ctx, mock_tool)
        assert "[DEDUP]" not in r1

        r2 = await tracing_toolset.call_tool("bash", args2, mock_ctx, mock_tool)
        assert "[DEDUP]" not in r2

    @pytest.mark.asyncio
    async def test_write_actions_never_deduplicated(
        self, tracing_toolset, mock_ctx, mock_tool
    ):
        """Write actions should always execute, even if repeated."""
        args = {"command": "python3 exploit.py"}

        r1 = await tracing_toolset.call_tool("bash", args, mock_ctx, mock_tool)
        assert "[DEDUP]" not in r1

        r2 = await tracing_toolset.call_tool("bash", args, mock_ctx, mock_tool)
        assert "[DEDUP]" not in r2


class TestTracingToolsetCacheInvalidation:
    """Integration tests for cache clearing after state-changing actions."""

    @pytest.mark.asyncio
    async def test_write_file_clears_action_cache(
        self, tracing_toolset, mock_ctx, mock_tool
    ):
        """After write_file, previously cached read-only actions should be gone."""
        state = tracing_toolset.state

        # Record a read-only action
        await tracing_toolset.call_tool(
            "bash", {"command": "cat /tmp/config.txt"}, mock_ctx, mock_tool
        )
        assert state.has_action("read:/tmp/config.txt")[0] is True

        # Now do a write_file — this should clear the cache via _update_state
        await tracing_toolset.call_tool(
            "write_file", {"path": "/workspace/shell.php", "content": "<?php"},
            mock_ctx, mock_tool
        )

        # Cache should be cleared
        assert state.has_action("read:/tmp/config.txt")[0] is False
        # But the write itself should be tracked in tried
        assert any("shell.php" in t for t in state.tried)

    @pytest.mark.asyncio
    async def test_nonreadonly_bash_clears_action_cache(
        self, tracing_toolset, mock_ctx, mock_tool
    ):
        """A non-readonly bash command (e.g. python3 script) should clear the cache."""
        state = tracing_toolset.state

        # Record read-only actions
        await tracing_toolset.call_tool(
            "bash", {"command": "curl http://target.com/"}, mock_ctx, mock_tool
        )
        await tracing_toolset.call_tool(
            "list_files", {"path": "/challenge/distfiles"}, mock_ctx, mock_tool
        )
        assert len(state._action_cache) == 2

        # Execute a non-readonly bash command
        await tracing_toolset.call_tool(
            "bash", {"command": "python3 exploit.py"}, mock_ctx, mock_tool
        )

        # Cache should be cleared
        assert len(state._action_cache) == 0

    @pytest.mark.asyncio
    async def test_readonly_bash_does_not_clear_cache(
        self, tracing_toolset, mock_ctx, mock_tool
    ):
        """A read-only bash command should NOT clear the action cache."""
        state = tracing_toolset.state

        # Record some read-only actions
        await tracing_toolset.call_tool(
            "bash", {"command": "file /challenge/distfiles/binary"}, mock_ctx, mock_tool
        )
        await tracing_toolset.call_tool(
            "bash", {"command": "curl http://target.com/"}, mock_ctx, mock_tool
        )
        assert len(state._action_cache) == 2

        # Execute another read-only bash command
        await tracing_toolset.call_tool(
            "bash", {"command": "cat /challenge/distfiles/readme.txt"}, mock_ctx, mock_tool
        )

        # Cache should still have all 3 entries
        assert len(state._action_cache) == 3


# ===========================================================================
# Integration: Solver.run_until_done_or_gave_up exception handling
# ===========================================================================


@pytest.fixture
def solver_instance(tmp_path):
    """Create a Solver with all heavy dependencies mocked.

    The solver is constructed but NOT started — we mock start() to set up
    _agent directly, avoiding all the heavy Docker/model resolution.
    """
    from unittest.mock import patch, MagicMock
    from backend.agents.solver import Solver
    from backend.cost_tracker import CostTracker
    from backend.sandbox import DockerSandbox
    from backend.ctfd import CTFdClient
    from backend.prompts import ChallengeMeta

    meta = ChallengeMeta(name="test_challenge", category="web")
    ctfd = MagicMock(spec=CTFdClient)
    cost_tracker = CostTracker()
    settings = MagicMock()
    settings.sandbox_image = "ctf-sandbox"
    settings.container_memory_limit = "4g"

    # Mock DockerSandbox
    sandbox = MagicMock(spec=DockerSandbox)
    sandbox._container = None
    sandbox.workspace_dir = "/workspace"
    sandbox.start = AsyncMock()
    sandbox.stop = AsyncMock()
    sandbox.exec = AsyncMock(return_value=MagicMock(stdout="x86_64"))

    # We need to patch resolve_model at the module level during Solver.__init__
    # because __init__ calls model_id_from_spec and provider_from_spec which
    # parse the model_spec string — these are fine with any string.
    # The issue is start() calling resolve_model(). We patch start() instead.
    with patch("backend.agents.solver.list_distfiles", return_value=[]):
        solver = Solver(
            model_spec="test-model",
            challenge_dir="/challenge/test",
            meta=meta,
            ctfd=ctfd,
            cost_tracker=cost_tracker,
            settings=settings,
            sandbox=sandbox,
        )

    # Replace start() with a version that just sets _agent to a mock
    async def fake_start():
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock()
        solver._agent = mock_agent

    solver.start = fake_start
    return solver


class TestUsageLimitExceededHandling:
    """Verify that UsageLimitExceeded inside run_until_done_or_gave_up returns GAVE_UP."""

    @pytest.mark.asyncio
    async def test_budget_exceeded_returns_gave_up(self, solver_instance):
        """When UsageLimitExceeded is raised, result status must be GAVE_UP, not ERROR."""
        from pydantic_ai.exceptions import UsageLimitExceeded
        from backend.solver_base import GAVE_UP, ERROR

        solver = solver_instance

        # Start the solver (uses the mocked deps)
        await solver.start()
        assert solver._agent is not None

        # Mock agent.run to raise UsageLimitExceeded
        solver._agent.run = AsyncMock(side_effect=UsageLimitExceeded("request limit reached"))

        result = await solver.run_until_done_or_gave_up()

        assert result.status == GAVE_UP, f"Expected GAVE_UP, got {result.status}"
        assert result.status != ERROR

    @pytest.mark.asyncio
    async def test_budget_exceeded_findings_mentions_budget(self, solver_instance):
        """The findings_summary should indicate budget exhaustion."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        solver = solver_instance
        await solver.start()
        solver._agent.run = AsyncMock(side_effect=UsageLimitExceeded("request limit reached"))

        result = await solver.run_until_done_or_gave_up()

        # Either "Budget exhausted" or "budget exhausted" should appear
        assert "udget" in result.findings_summary.lower() or "budget" in result.findings_summary.lower()

    @pytest.mark.asyncio
    async def test_budget_exceeded_step_count_reflects_work(self, solver_instance):
        """Step count should reflect work done, even when budget ran out."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        solver = solver_instance
        solver._step_count = [7]  # Simulate 7 tool calls before budget ran out
        await solver.start()
        solver._agent.run = AsyncMock(side_effect=UsageLimitExceeded("limit"))

        result = await solver.run_until_done_or_gave_up()
        assert result.step_count == 7


class TestConfirmedFlagPreservation:
    """Verify deps.confirmed_flag is preserved across all exception paths."""

    @pytest.mark.asyncio
    async def test_budget_exceeded_preserves_confirmed_flag(self, solver_instance):
        """UsageLimitExceeded with deps.confirmed_flag set → flag preserved, confirmed=True."""
        from pydantic_ai.exceptions import UsageLimitExceeded
        from backend.solver_base import GAVE_UP

        solver = solver_instance
        solver.deps.confirmed_flag = "flag{confirmed_via_ctfd}"
        await solver.start()
        solver._agent.run = AsyncMock(side_effect=UsageLimitExceeded("limit"))

        result = await solver.run_until_done_or_gave_up()

        assert result.status == GAVE_UP
        assert result.flag == "flag{confirmed_via_ctfd}"
        assert solver._confirmed is True

    @pytest.mark.asyncio
    async def test_cancelled_preserves_confirmed_flag(self, solver_instance):
        """asyncio.CancelledError with deps.confirmed_flag set → flag preserved."""
        from backend.solver_base import CANCELLED

        solver = solver_instance
        solver.deps.confirmed_flag = "flag{confirmed_via_ctfd}"
        await solver.start()
        solver._agent.run = AsyncMock(side_effect=asyncio.CancelledError())

        result = await solver.run_until_done_or_gave_up()

        assert result.status == CANCELLED
        assert result.flag == "flag{confirmed_via_ctfd}"
        assert solver._confirmed is True

    @pytest.mark.asyncio
    async def test_generic_exception_preserves_confirmed_flag(self, solver_instance):
        """Generic Exception with deps.confirmed_flag set → flag preserved, status=ERROR."""
        from backend.solver_base import ERROR

        solver = solver_instance
        solver.deps.confirmed_flag = "flag{confirmed_via_ctfd}"
        await solver.start()
        solver._agent.run = AsyncMock(side_effect=RuntimeError("something broke"))

        result = await solver.run_until_done_or_gave_up()

        assert result.status == ERROR
        assert result.flag == "flag{confirmed_via_ctfd}"
        assert solver._confirmed is True

    @pytest.mark.asyncio
    async def test_budget_exceeded_without_confirmed_flag_no_false_positive(self, solver_instance):
        """UsageLimitExceeded without deps.confirmed_flag → no flag hallucinated."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        solver = solver_instance
        # No confirmed_flag, no flag_candidates
        await solver.start()
        solver._agent.run = AsyncMock(side_effect=UsageLimitExceeded("limit"))

        result = await solver.run_until_done_or_gave_up()

        assert result.flag is None

    @pytest.mark.asyncio
    async def test_budget_exceeded_confirmed_overrides_candidate(self, solver_instance):
        """When both confirmed_flag and flag_candidates exist, confirmed wins."""
        from pydantic_ai.exceptions import UsageLimitExceeded

        solver = solver_instance
        solver.deps.confirmed_flag = "flag{confirmed}"
        solver.investigation.flag_candidates.append("flag{candidate}")
        await solver.start()
        solver._agent.run = AsyncMock(side_effect=UsageLimitExceeded("limit"))

        result = await solver.run_until_done_or_gave_up()

        # The code does: self._flag = self._flag or self.deps.confirmed_flag
        # Then if not confirmed, checks candidates. Since confirmed_flag is set,
        # _confirmed=True, so it returns GAVE_UP with the confirmed flag.
        assert result.flag == "flag{confirmed}"

    @pytest.mark.asyncio
    async def test_cancelled_preserves_flag_candidate(self, solver_instance):
        """asyncio.CancelledError with flag_candidate but no confirmed_flag → candidate used."""
        from backend.solver_base import CANCELLED

        solver = solver_instance
        solver.investigation.flag_candidates.append("flag{best_candidate}")
        await solver.start()
        solver._agent.run = AsyncMock(side_effect=asyncio.CancelledError())

        result = await solver.run_until_done_or_gave_up()

        assert result.status == CANCELLED
        assert result.flag == "flag{best_candidate}"
        assert "cancelled" in result.findings_summary.lower()

    @pytest.mark.asyncio
    async def test_generic_exception_preserves_flag_candidate(self, solver_instance):
        """Generic Exception with flag_candidate but no confirmed_flag → candidate used."""
        from backend.solver_base import ERROR

        solver = solver_instance
        solver.investigation.flag_candidates.append("flag{error_candidate}")
        await solver.start()
        solver._agent.run = AsyncMock(side_effect=RuntimeError("boom"))

        result = await solver.run_until_done_or_gave_up()

        assert result.status == ERROR
        assert result.flag == "flag{error_candidate}"
        assert "error" in result.findings_summary.lower()
