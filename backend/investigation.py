"""Pure investigation logic — no framework dependencies.

This module contains all deterministic functions and the InvestigationState
dataclass used by the solver. Separating them allows testing without
importing pydantic_ai, boto3, or other heavy dependencies.

All solver-specific integration lives in backend/agents/solver.py.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REQUEST_BUDGET = 25

FLAG_PATTERNS = [
    re.compile(r'(?:flag|FLAG|ctf|CTF|picoCTF|PicoCTF)\{[^}]+\}'),
    re.compile(r'(?i)flag[:\s=]+([A-Za-z0-9_{}\-]{12,})'),
]

_READONLY_BASH_COMMANDS = frozenset({
    "cat", "head", "tail", "less", "more", "file", "strings",
    "readelf", "objdump", "nm", "xxd", "hexdump", "ls", "find",
    "tree", "grep", "rg", "nmap", "exiftool", "pngcheck",
    "identify", "vol", "fls", "mmls", "fsstat", "check",
    "curl", "wget",
})

_READONLY_TOOLS = frozenset({"list_files", "read_file", "view_image", "webhook_get_requests"})

_CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "web": {"web", "http", "https", "url", "upload", "xss", "sqli", "ssrf",
            "lfi", "rfi", "rce", "sql", "injection", "csrf", "ssti",
            "php", "javascript", "html", "cookie", "session"},
    "crypto": {"crypto", "rsa", "aes", "cipher", "encrypt", "decrypt",
               "key", "nonce", "hash", "signature", "certificate", "openssl"},
    "reverse": {"reverse", "reversing", "re", "binary", "disasm", "decompil",
                "ghidra", "ida", "radare"},
    "pwn": {"pwn", "exploit", "buffer", "overflow", "rop", "shellcode",
            "heap", "stack", "format string", "got", "plt"},
    "forensics": {"forensic", "forensics", "disk", "memory", "pcap",
                  "volatility", "filesystem", "carving", "sleuthkit"},
    "steganography": {"steg", "steganography", "lsb", "hidden", "embed",
                      "exif", "zsteg", "steghide"},
    "osint": {"osint", "geolocate", "geolocation", "social", "recon",
              "shodan", "dns"},
    "misc": {"misc", "programming", "script", "algorithm", "math",
             "logic", "encoding"},
}

# ---------------------------------------------------------------------------
# Investigation State
# ---------------------------------------------------------------------------


@dataclass
class InvestigationState:
    """Compact investigation state — the solver's working memory."""

    category: str = ""
    target: str = ""
    hypothesis: str = ""
    objective: str = ""
    discoveries: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)
    tried: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    uploaded_files: list[str] = field(default_factory=list)
    flag_candidates: list[str] = field(default_factory=list)
    progress_notes: str = ""
    # Internal tracking (not rendered in context)
    _action_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _max_cache: int = 50
    _no_progress_streak: int = 0

    def add_discovery(self, fact: str) -> bool:
        """Add a discovery. Returns True if it is genuinely new."""
        fact = fact.strip()[:200]
        if not fact or fact in self.discoveries:
            return False
        self.discoveries.append(fact)
        self._no_progress_streak = 0
        return True

    def add_endpoint(self, url: str) -> bool:
        url = url.strip().rstrip("/")
        if url and url not in self.endpoints:
            self.endpoints.append(url)
            return True
        return False

    def record_action(self, normalized: str, result_summary: str) -> None:
        """Record a normalized action for deduplication."""
        while len(self._action_cache) >= self._max_cache:
            self._action_cache.popitem(last=False)
        self._action_cache[normalized] = result_summary[:300]

    def has_action(self, normalized: str) -> tuple[bool, str]:
        """Check if an equivalent action was already performed.
        Returns (found, prior_result_summary)."""
        prior = self._action_cache.get(normalized)
        return (prior is not None, prior or "")

    def increment_no_progress(self) -> int:
        self._no_progress_streak += 1
        return self._no_progress_streak

    @property
    def no_progress_streak(self) -> int:
        return self._no_progress_streak


# ---------------------------------------------------------------------------
# Challenge classification
# ---------------------------------------------------------------------------


def classify_challenge(meta: Any, distfile_names: list[str]) -> str:
    """Deterministically classify a challenge from metadata. No LLM needed.

    `meta` must have: category, tags, connection_info, description attributes.
    """
    # 1. Explicit category from CTFd
    cat = (getattr(meta, "category", "") or "").lower().strip()
    if cat in _CATEGORY_KEYWORDS:
        return cat

    # 2. Tags
    for tag in getattr(meta, "tags", []) or []:
        tag_lower = tag.lower()
        for category, keywords in _CATEGORY_KEYWORDS.items():
            if tag_lower in keywords or tag_lower == category:
                return category

    # 3. Connection info
    conn = (getattr(meta, "connection_info", "") or "").lower()
    if re.match(r"^https?://", conn) or "curl" in conn:
        return "web"
    if conn.startswith("nc "):
        desc_lower = (getattr(meta, "description", "") or "").lower()
        if any(kw in desc_lower for kw in _CATEGORY_KEYWORDS["pwn"]):
            return "pwn"
        if any(kw in desc_lower for kw in _CATEGORY_KEYWORDS["crypto"]):
            return "crypto"
        return "pwn"

    # 4. Description keyword scoring
    desc_lower = (getattr(meta, "description", "") or "").lower()
    scores: dict[str, int] = {c: 0 for c in _CATEGORY_KEYWORDS}
    for word in re.findall(r"\b\w+\b", desc_lower):
        for category, keywords in _CATEGORY_KEYWORDS.items():
            if word in keywords:
                scores[category] += 1
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] > 0:
        return best

    # 5. Distfile extensions (only when metadata is uninformative)
    for name in distfile_names:
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext in ("php", "html", "js", "asp", "aspx", "jsp"):
            return "web"
        if ext in ("exe", "elf", "bin", "o", "so"):
            return "reverse"
        if ext in ("pcap", "mem", "img", "raw", "dd", "e01"):
            return "forensics"
        if ext in ("png", "jpg", "jpeg", "gif", "bmp", "wav", "mp3"):
            return "steganography"
        if ext in ("py", "c", "cpp", "java", "rs"):
            return "misc"

    return "misc"


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------


def extract_target(conn: str) -> str:
    """Extract a useful target string from connection_info."""
    if not conn:
        return ""
    conn = conn.strip()
    m = re.match(r"(https?://[^\s]+)", conn)
    if m:
        return m.group(1).rstrip("/")
    m = re.match(r"nc\s+(\S+(?:\s+\S+)?)", conn)
    if m:
        return m.group(1).strip()
    return conn[:100]


# ---------------------------------------------------------------------------
# URL / action normalization
# ---------------------------------------------------------------------------


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    url = url.strip().rstrip("/")
    return re.sub(r"\?.*$", "", url)


def is_readonly_action(tool_name: str, args: dict[str, Any]) -> bool:
    """Determine if a tool action is read-only (safe to dedup)."""
    if tool_name in _READONLY_TOOLS:
        return True
    if tool_name == "web_fetch" and args.get("method", "GET").upper() == "GET":
        return True
    if tool_name == "bash":
        cmd = args.get("command", "").strip()
        parts = cmd.split()
        if not parts:
            return True
        first_word = parts[0]
        if first_word in _READONLY_BASH_COMMANDS:
            # curl/wget with data payload is NOT read-only
            if first_word in ("curl", "wget"):
                cmd_lower = cmd.lower()
                if any(kw in cmd_lower for kw in ("-d ", "--data", "-x post", "--request post")):
                    return False
            return True
    return False


def normalize_action(tool_name: str, args: dict[str, Any]) -> str:
    """Normalize a tool call signature for deduplication."""
    if tool_name == "bash":
        cmd = args.get("command", "").strip()
        first_line = cmd.split("\n")[0].strip()
        parts = first_line.split()
        if not parts:
            return f"bash:{cmd[:80]}"

        cmd_name = parts[0]

        # curl: extract method + URL
        if cmd_name == "curl":
            method = "GET"
            url = ""
            i = 1
            while i < len(parts):
                p = parts[i]
                if p in ("-X", "--request") and i + 1 < len(parts):
                    method = parts[i + 1].upper()
                    i += 2
                    continue
                if p in ("-d", "--data", "--data-raw", "--data-binary") and i + 1 < len(parts):
                    method = "POST"
                    i += 2
                    continue
                if not p.startswith("-") and ("http" in p or "://" in p
                                           or (p.startswith("/") and not url)):
                    url = normalize_url(p)
                i += 1
            return f"curl:{method}:{url or '?'}"

        # File inspection tools: normalize by target file
        if cmd_name in ("file", "strings", "readelf", "objdump", "nm", "xxd", "hexdump"):
            target = ""
            for p in parts[1:]:
                if not p.startswith("-"):
                    # Skip numeric values that are flag arguments (e.g., -n 8)
                    if target == "" and p.isdigit():
                        continue
                    target = p
                    break
            return f"inspect:{cmd_name}:{target or '?'}"

        # Directory listing
        if cmd_name in ("ls", "find", "tree"):
            for p in parts[1:4]:
                if not p.startswith("-") and not p.startswith("|"):
                    return f"list:{p}"
            return "list:."

        # File reading
        if cmd_name in ("cat", "head", "tail", "less", "more"):
            for p in parts[1:]:
                if not p.startswith("-"):
                    return f"read:{p}"
            return "read:?"

        # Script execution
        if cmd_name in ("python3", "python", "sage"):
            rest = " ".join(parts[1:])[:80]
            return f"script:{cmd_name}:{rest[:40]}"

        # Fallback: command name + first non-flag arg
        first_arg = ""
        for p in parts[1:]:
            if not p.startswith("-") and not p.startswith("|"):
                first_arg = p
                break
        return f"bash:{cmd_name}:{first_arg}"

    if tool_name == "web_fetch":
        url = normalize_url(args.get("url", ""))
        method = args.get("method", "GET").upper()
        return f"web_fetch:{method}:{url}"
    if tool_name == "list_files":
        return f"list_files:{args.get('path', '/challenge/distfiles')}"
    if tool_name == "read_file":
        return f"read_file:{args.get('path', '')}"
    if tool_name == "write_file":
        path = args.get("path", "")
        return f"write_file:{path.rsplit('/', 1)[-1] if '/' in path else path}"
    if tool_name == "view_image":
        return f"view_image:{args.get('filename', '')}"
    if tool_name == "submit_flag":
        return f"submit_flag:{args.get('flag', '')}"
    return f"{tool_name}"


# ---------------------------------------------------------------------------
# Flag detection
# ---------------------------------------------------------------------------


def detect_flag_in_text(text: str) -> list[str]:
    """Scan text for likely CTF flags. Returns list of candidates."""
    candidates: list[str] = []
    for pattern in FLAG_PATTERNS:
        for match in pattern.finditer(text):
            candidate = match.group(0).strip()
            # Filter obvious non-flags
            if candidate.lower() in ("flag:", "flag", "flags",
                                      "flag{flag}", "flag{placeholder}"):
                continue
            if len(candidate) < 6:
                continue
            candidates.append(candidate)
    return candidates


# ---------------------------------------------------------------------------
# State context rendering
# ---------------------------------------------------------------------------


def build_state_context(state: InvestigationState, remaining_budget: int) -> str:
    """Render investigation state as a compact context block."""
    lines = ["## Investigation State"]
    if state.category:
        lines.append(f"- **Category**: {state.category}")
    if state.target:
        lines.append(f"- **Target**: {state.target}")
    if state.hypothesis:
        lines.append(f"- **Hypothesis**: {state.hypothesis}")
    if state.objective:
        lines.append(f"- **Objective**: {state.objective}")
    if state.endpoints:
        lines.append(f"- **Endpoints**: {', '.join(state.endpoints[-8:])}")
    if state.technologies:
        lines.append(f"- **Tech**: {', '.join(state.technologies[-6:])}")
    if state.discoveries:
        recent = state.discoveries[-6:]
        lines.append("- **Discoveries**:")
        for d in recent:
            lines.append(f"  - {d}")
    if state.failed:
        lines.append(f"- **Failed**: {', '.join(state.failed[-5:])}")
    if state.uploaded_files:
        lines.append(f"- **Uploaded**: {', '.join(state.uploaded_files[-4:])}")
    if state.flag_candidates:
        lines.append(f"- **Flag candidates**: {', '.join(state.flag_candidates)}")
    lines.append(f"- **Budget**: ~{remaining_budget} requests remaining")
    if state.no_progress_streak >= 3:
        lines.append(
            f"- **NO PROGRESS** for {state.no_progress_streak} actions. Change strategy."
        )
    return "\n".join(lines)
