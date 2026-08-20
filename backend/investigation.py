"""Compact investigation state for the CTF solver.

Deterministic orchestration lives here so the LLM spends requests on
strategy, not on repeating work or re-deriving context after compaction.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse, urlunparse


STUCK_THRESHOLD = 4
MAX_FACTS = 40
MAX_ACTIONS = 80
MAX_LAST_ACTIONS = 8
SUMMARY_TOOL_CHARS = 1200

URL_RE = re.compile(r"https?://[^\s)\]>'\"<]+", re.IGNORECASE)
FLAG_RE = re.compile(r"\b[A-Za-z0-9_]{2,32}\{[^{}\r\n]{1,256}\}")
PLACEHOLDER_FLAGS = {
    "ctf{flag}",
    "ctf{placeholder}",
    "flag{flag}",
    "picoctf{flag}",
}

FILE_CATEGORIES = frozenset(
    {
        "reversing",
        "pwn",
        "crypto",
        "forensics",
        "steganography",
        "misc",
    }
)

WEB_HINTS = (
    "website",
    "web page",
    "webpage",
    "upload",
    "http",
    "php",
    "cookie",
    "ssti",
    "xss",
    "sql injection",
    "lfi",
    "rfi",
    "ssrf",
    "robots.txt",
    "file upload",
    "profile picture",
)

STALL_MESSAGE = (
    "INVESTIGATION STALLED.\n"
    "The last actions produced no new information.\n"
    "Do not repeat the current approach.\n"
    "Review known facts, successful techniques, failed techniques, "
    "current privileges, and challenge category.\n"
    "Generate several alternative hypotheses and select the highest-value "
    "unexplored one."
)


class ChallengeCategory(StrEnum):
    WEB = "web"
    PWN = "pwn"
    REVERSING = "reversing"
    CRYPTO = "crypto"
    FORENSICS = "forensics"
    STEGANOGRAPHY = "steganography"
    OSINT = "osint"
    MISC = "misc"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    NETWORK = "network"
    UNKNOWN = "unknown"


class InvestigationPhase(StrEnum):
    RECON = "RECON"
    DISCOVERY = "DISCOVERY"
    VULNERABILITY_IDENTIFICATION = "VULNERABILITY_IDENTIFICATION"
    EXPLOITATION = "EXPLOITATION"
    POST_EXPLOITATION = "POST_EXPLOITATION"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    FLAG_SEARCH = "FLAG_SEARCH"
    FLAG_VERIFICATION = "FLAG_VERIFICATION"
    COMPLETED = "COMPLETED"
    STUCK = "STUCK"


class HypothesisStatus(StrEnum):
    UNTESTED = "UNTESTED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


_CATEGORY_ALIASES: dict[str, ChallengeCategory] = {
    "web": ChallengeCategory.WEB,
    "pwn": ChallengeCategory.PWN,
    "binary": ChallengeCategory.PWN,
    "reverse": ChallengeCategory.REVERSING,
    "reversing": ChallengeCategory.REVERSING,
    "re": ChallengeCategory.REVERSING,
    "rev": ChallengeCategory.REVERSING,
    "crypto": ChallengeCategory.CRYPTO,
    "cryptography": ChallengeCategory.CRYPTO,
    "forensics": ChallengeCategory.FORENSICS,
    "forensic": ChallengeCategory.FORENSICS,
    "stego": ChallengeCategory.STEGANOGRAPHY,
    "steganography": ChallengeCategory.STEGANOGRAPHY,
    "osint": ChallengeCategory.OSINT,
    "misc": ChallengeCategory.MISC,
    "privesc": ChallengeCategory.PRIVILEGE_ESCALATION,
    "privilege escalation": ChallengeCategory.PRIVILEGE_ESCALATION,
    "network": ChallengeCategory.NETWORK,
}

_PHASE_OBJECTIVES: dict[InvestigationPhase, str] = {
    InvestigationPhase.RECON: "Understand the target and attack surface.",
    InvestigationPhase.DISCOVERY: "Identify endpoints, files, and input surfaces.",
    InvestigationPhase.VULNERABILITY_IDENTIFICATION: "Confirm a useful vulnerability.",
    InvestigationPhase.EXPLOITATION: "Turn a discovered vulnerability into useful access.",
    InvestigationPhase.POST_EXPLOITATION: "Determine execution context and reachable data.",
    InvestigationPhase.PRIVILEGE_ESCALATION: (
        "Find an alternate path to protected data or escalate privileges."
    ),
    InvestigationPhase.FLAG_SEARCH: "Locate a candidate flag.",
    InvestigationPhase.FLAG_VERIFICATION: "Confirm the candidate flag is real.",
    InvestigationPhase.COMPLETED: "Return the verified flag and stop.",
    InvestigationPhase.STUCK: "Reassess strategy; do not repeat failed approaches.",
}


@dataclass
class Hypothesis:
    statement: str
    status: HypothesisStatus = HypothesisStatus.UNTESTED


@dataclass
class RecordedAction:
    signature: str
    tool: str
    privilege_generation: int
    note: str = ""


@dataclass
class InvestigationState:
    challenge_description: str = ""
    hint: str = ""
    evidence_paths: list[str] = field(default_factory=list)
    target_urls: list[str] = field(default_factory=list)
    category: ChallengeCategory = ChallengeCategory.UNKNOWN
    phase: InvestigationPhase = InvestigationPhase.RECON
    current_objective: str = _PHASE_OBJECTIVES[InvestigationPhase.RECON]
    known_facts: list[str] = field(default_factory=list)
    discoveries: list[str] = field(default_factory=list)
    successful_actions: list[str] = field(default_factory=list)
    failed_actions: list[str] = field(default_factory=list)
    important_endpoints: list[str] = field(default_factory=list)
    credentials: list[str] = field(default_factory=list)
    execution_context: dict[str, Any] = field(default_factory=dict)
    privileges: dict[str, Any] = field(default_factory=dict)
    files_of_interest: list[str] = field(default_factory=list)
    services_of_interest: list[str] = field(default_factory=list)
    candidate_flags: list[str] = field(default_factory=list)
    confirmed_flag: str | None = None
    hypotheses: list[Hypothesis] = field(default_factory=list)
    retry_count: int = 0
    consecutive_no_progress: int = 0
    last_actions: list[str] = field(default_factory=list)
    next_best_action: str = ""
    privilege_generation: int = 0
    recorded_actions: list[RecordedAction] = field(default_factory=list)
    last_full_results: dict[str, str] = field(default_factory=dict)

    def add_fact(self, fact: str) -> bool:
        fact = fact.strip()
        if not fact or fact in self.known_facts:
            return False
        self.known_facts.append(fact)
        if len(self.known_facts) > MAX_FACTS:
            self.known_facts = self.known_facts[-MAX_FACTS:]
        return True

    def add_discovery(self, item: str) -> bool:
        item = item.strip()
        if not item or item in self.discoveries:
            return False
        self.discoveries.append(item)
        return True

    def set_phase(self, phase: InvestigationPhase, objective: str | None = None) -> None:
        if self.phase == InvestigationPhase.COMPLETED:
            return
        self.phase = phase
        self.current_objective = objective or _PHASE_OBJECTIVES[phase]

    def set_hypothesis(self, statement: str, status: HypothesisStatus) -> None:
        for hyp in self.hypotheses:
            if hyp.statement == statement:
                hyp.status = status
                return
        self.hypotheses.append(Hypothesis(statement, status))

    def rejected_statements(self) -> list[str]:
        return [h.statement for h in self.hypotheses if h.status == HypothesisStatus.REJECTED]

    def confirmed_statements(self) -> list[str]:
        return [h.statement for h in self.hypotheses if h.status == HypothesisStatus.CONFIRMED]


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    for match in URL_RE.findall(text):
        url = match.rstrip(".,;:)")
        if url not in found:
            found.append(url)
    return found


def classify_challenge(
    description: str,
    hint: str = "",
    evidence_paths: list[str] | None = None,
    metadata_category: str = "",
    challenge_name: str = "",
) -> ChallengeCategory:
    """Classify the challenge to set investigation priorities."""

    blob = f"{description} {hint}".lower()
    urls = extract_urls(description) + extract_urls(hint)
    evidence_paths = evidence_paths or []

    if urls and any(token in blob for token in WEB_HINTS):
        return ChallengeCategory.WEB
    if urls:
        return ChallengeCategory.WEB

    interactive = challenge_name.lower() == "interactive"
    mapped = _CATEGORY_ALIASES.get(metadata_category.strip().lower())
    if mapped and mapped != ChallengeCategory.UNKNOWN and not interactive:
        return mapped

    keyword_map: list[tuple[tuple[str, ...], ChallengeCategory]] = [
        (("pwn", "buffer overflow", "rop", "shellcode", "libc"), ChallengeCategory.PWN),
        (("revers", "disassembl", "decompil", "ghidra", "binary"), ChallengeCategory.REVERSING),
        (("cipher", "encrypt", "rsa", "aes", "xor", "crypto"), ChallengeCategory.CRYPTO),
        (("pcap", "memory dump", "disk image", "forensic", "wireshark"), ChallengeCategory.FORENSICS),
        (("steg", "hidden in image", "lsb"), ChallengeCategory.STEGANOGRAPHY),
        (("osint", "whois", "shodan"), ChallengeCategory.OSINT),
        (("privesc", "privilege", "suid", "sudo"), ChallengeCategory.PRIVILEGE_ESCALATION),
        (("upload", "website", "http", "cookie", "ssti"), ChallengeCategory.WEB),
    ]
    for tokens, category in keyword_map:
        if any(token in blob for token in tokens):
            return category

    joined_evidence = " ".join(evidence_paths).lower()
    if any(ext in joined_evidence for ext in (".pcap", ".mem", ".raw", ".dd", ".e01")):
        return ChallengeCategory.FORENSICS
    if any(ext in joined_evidence for ext in (".exe", ".elf", ".bin")):
        return ChallengeCategory.REVERSING

    if mapped:
        return mapped
    return ChallengeCategory.UNKNOWN


def should_inspect_distfiles(state: InvestigationState) -> bool:
    """Local distfiles are a priority only when they are the attack surface."""

    if state.evidence_paths:
        return True
    if state.category in FILE_CATEGORIES and not state.target_urls:
        return True
    if state.consecutive_no_progress >= STUCK_THRESHOLD:
        return True
    return False


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        return url.strip().rstrip("/")
    path = parsed.path.rstrip("/") or "/"
    netloc = parsed.netloc.lower()
    return urlunparse((parsed.scheme.lower(), netloc, path, "", parsed.query, ""))


def payload_family(content: str) -> str:
    compact = re.sub(r"\s+", "", content.lower())
    if "<?php" in compact or compact.startswith("<?="):
        if "whoami" in compact:
            return "php:whoami"
        if "/root" in compact:
            return "php:read_root"
        if any(fn in compact for fn in ("system(", "passthru(", "shell_exec(", "exec(")):
            return "php:cmd_exec"
        if "phpinfo" in compact:
            return "php:phpinfo"
        return "php:generic"
    digest = hashlib.sha256(compact.encode()).hexdigest()[:16]
    return f"content:{digest}"


def _normalize_command(command: str) -> str:
    return " ".join(command.strip().split())


def action_signature(tool: str, args: dict[str, Any] | None) -> str:
    """Normalize a tool call into a comparable signature."""

    args = args or {}
    name = tool.strip().lower()

    if name == "web_fetch":
        method = str(args.get("method") or "GET").upper()
        url = normalize_url(str(args.get("url") or ""))
        body = str(args.get("body") or "")
        extra = payload_family(body) if body else ""
        return f"http:{method}:{url}:{extra}"

    if name == "bash":
        command = _normalize_command(str(args.get("command") or ""))
        curl = re.match(
            r"^(?:curl|wget)\s+(?:-[A-Za-z]+\s+)*['\"]?(https?://\S+)",
            command,
            re.IGNORECASE,
        )
        if curl:
            return f"http:GET:{normalize_url(curl.group(1).rstrip(chr(39)))}:"
        php_write = re.search(r"(?:cat|tee)\s+.*\.(php|phtml)", command, re.IGNORECASE)
        if php_write or "<?php" in command.lower():
            return f"upload:{payload_family(command)}"
        return f"bash:{command}"

    if name == "write_file":
        path = str(args.get("path") or "")
        content = str(args.get("content") or "")
        return f"write:{path}:{payload_family(content)}"

    if name == "read_file":
        return f"read:{args.get('path', '')}"

    if name == "list_files":
        return f"list:{args.get('path', '/challenge/distfiles')}"

    if name == "inspect_distfiles":
        return "inspect_distfiles"

    if name == "submit_flag":
        return f"submit:{args.get('flag', '')}"

    raw = hashlib.sha256(repr(sorted(args.items())).encode()).hexdigest()[:16]
    return f"{name}:{raw}"


class ActionTracker:
    """Remember attempted actions and detect duplicates."""

    def is_duplicate(self, state: InvestigationState, signature: str) -> bool:
        for recorded in state.recorded_actions:
            if recorded.signature != signature:
                continue
            if recorded.privilege_generation < state.privilege_generation:
                continue
            if state.phase == InvestigationPhase.FLAG_VERIFICATION:
                continue
            return True
        return False

    def record(self, state: InvestigationState, tool: str, signature: str, note: str = "") -> None:
        state.recorded_actions.append(
            RecordedAction(
                signature=signature,
                tool=tool,
                privilege_generation=state.privilege_generation,
                note=note,
            )
        )
        if len(state.recorded_actions) > MAX_ACTIONS:
            state.recorded_actions = state.recorded_actions[-MAX_ACTIONS:]
        state.last_actions.append(signature)
        if len(state.last_actions) > MAX_LAST_ACTIONS:
            state.last_actions = state.last_actions[-MAX_LAST_ACTIONS:]


def extract_flag_candidate(text: str) -> str | None:
    if not text:
        return None
    match = FLAG_RE.search(text)
    if not match:
        return None
    candidate = match.group(0).strip()
    if len(candidate) > 300:
        return None
    if candidate.lower() in PLACEHOLDER_FLAGS:
        return None
    inner = candidate[candidate.find("{") + 1 : -1].strip().lower()
    if inner in {"flag", "placeholder", "..."}:
        return None
    return candidate


def summarize_tool_result(tool: str, args: dict[str, Any] | None, result: str) -> str:
    """Compact observation for the model; full text is stored separately."""

    text = result or ""
    if len(text) <= SUMMARY_TOOL_CHARS:
        return text

    args = args or {}
    lines = [line for line in text.splitlines() if line.strip()]
    head = "\n".join(lines[:12])
    http = re.search(r"HTTP\s+(\d{3})", text)
    parts = [f"SUMMARY of {tool}:"]
    if http:
        parts.append(f"HTTP {http.group(1)}")
    url = args.get("url")
    if url:
        parts.append(f"Endpoint: {url}")
    if "permission denied" in text.lower():
        parts.append("Permission denied: yes")
    who = _extract_whoami(text, str(args.get("command") or ""))
    if who:
        parts.append(f"Execution user: {who}")
    parts.append(head[:800])
    parts.append(f"... [{len(text)} chars omitted; key observations retained]")
    return "\n".join(parts)


_KNOWN_USERS = frozenset({"www-data", "root", "apache", "nginx", "nobody", "kali"})


def _extract_whoami(text: str, command: str = "") -> str | None:
    uid = re.search(r"\buid=\d+\(([^)]+)\)", text)
    if uid:
        return uid.group(1).lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    asked = "whoami" in command.lower()
    if asked or (len(lines) <= 3):
        for line in reversed(lines):
            token = line.split()[-1].lower() if line.split() else ""
            if token in _KNOWN_USERS:
                return token
    return None


def apply_tool_result(
    state: InvestigationState,
    tool: str,
    args: dict[str, Any] | None,
    result: str,
) -> bool:
    """Update investigation state from a tool result.

    Returns True when meaningful new information appeared.
    """

    args = args or {}
    result = result or ""
    progress = False
    lower = result.lower()

    if tool == "web_fetch":
        url = str(args.get("url") or "")
        if url:
            progress |= state.add_discovery(f"fetched {normalize_url(url)}")
            if url not in state.important_endpoints:
                state.important_endpoints.append(url)
        if state.phase == InvestigationPhase.RECON:
            state.set_phase(InvestigationPhase.DISCOVERY)
        if "upload" in lower:
            progress |= state.add_fact("Upload functionality present in HTTP response.")
            state.set_phase(InvestigationPhase.VULNERABILITY_IDENTIFICATION)

    if tool == "inspect_distfiles":
        progress |= state.add_fact("Inspected /challenge/distfiles.")

    if "upload" in lower and (".php" in lower or "multipart" in lower):
        progress |= state.add_fact("Upload endpoint interaction observed.")
        if "upload.php" in lower and "upload.php" not in state.important_endpoints:
            state.important_endpoints.append("upload.php")

    if "<?php" in str(args.get("content") or "") or "<?php" in str(args.get("command") or ""):
        progress |= state.add_fact("Attempted PHP payload upload/write.")
        state.set_hypothesis("Uploaded PHP executes server-side.", HypothesisStatus.UNTESTED)

    who = _extract_whoami(result, str(args.get("command") or ""))
    if who:
        previous = state.privileges.get("execution_user")
        if previous != who:
            state.privileges["execution_user"] = who
            state.execution_context["execution_user"] = who
            state.privilege_generation += 1
            progress |= state.add_fact(f"Execution user: {who}")
            if who != "root":
                state.set_phase(
                    InvestigationPhase.POST_EXPLOITATION,
                    "Determine privileges and reachable data.",
                )

    if any(token in lower for token in ("uid=", "www-data", "php execution")):
        if "uid=" in lower or who == "www-data":
            state.execution_context["web_rce"] = True
            progress |= state.add_fact("Code execution confirmed.")
            state.set_hypothesis("Uploaded PHP executes server-side.", HypothesisStatus.CONFIRMED)
            if state.phase in {
                InvestigationPhase.RECON,
                InvestigationPhase.DISCOVERY,
                InvestigationPhase.VULNERABILITY_IDENTIFICATION,
                InvestigationPhase.EXPLOITATION,
            }:
                state.set_phase(InvestigationPhase.POST_EXPLOITATION)

    if "permission denied" in lower:
        progress |= state.add_fact("Permission denied on a protected path.")
        state.set_hypothesis("Direct access to /root is possible.", HypothesisStatus.REJECTED)
        state.privileges["direct_root_access"] = False
        if state.execution_context.get("web_rce") or state.privileges.get("execution_user"):
            state.set_phase(InvestigationPhase.PRIVILEGE_ESCALATION)
            state.next_best_action = (
                "Enumerate sudo, SUID, capabilities, cron, secrets, and local services."
            )

    candidate = extract_flag_candidate(result)
    if candidate and candidate not in state.candidate_flags:
        state.candidate_flags.append(candidate)
        progress |= state.add_fact(f"Candidate flag: {candidate}")
        state.set_phase(InvestigationPhase.FLAG_VERIFICATION)
        progress = True

    if "flag confirmed" in lower:
        state.set_phase(InvestigationPhase.COMPLETED)
        progress = True

    failed = (
        "permission denied" in lower
        or "not found" in lower
        or "error" in lower[:200]
        or "[exit" in lower
    )
    sig = action_signature(tool, args)
    if failed:
        if sig not in state.failed_actions:
            state.failed_actions.append(sig)
    elif result.strip() and result.strip() != "(no output)":
        if sig not in state.successful_actions:
            state.successful_actions.append(sig)

    if progress:
        state.consecutive_no_progress = 0
        if state.phase == InvestigationPhase.STUCK:
            state.set_phase(InvestigationPhase.PRIVILEGE_ESCALATION)
    else:
        state.consecutive_no_progress += 1
        if state.consecutive_no_progress >= STUCK_THRESHOLD:
            state.set_phase(InvestigationPhase.STUCK)

    return progress


def mark_confirmed_flag(state: InvestigationState, flag: str) -> None:
    state.confirmed_flag = flag
    if flag not in state.candidate_flags:
        state.candidate_flags.append(flag)
    state.set_phase(InvestigationPhase.COMPLETED, "Return the verified flag and stop.")
    state.add_fact(f"Confirmed flag: {flag}")
    state.consecutive_no_progress = 0


def compact_summary(state: InvestigationState) -> str:
    """Investigation summary used after compaction / retry."""

    rejected = state.rejected_statements() or ["(none)"]
    confirmed = state.confirmed_statements() or ["(none)"]
    unexplored = [
        h.statement for h in state.hypotheses if h.status == HypothesisStatus.UNTESTED
    ]
    if state.phase == InvestigationPhase.PRIVILEGE_ESCALATION and not unexplored:
        unexplored = [
            "sudo permissions",
            "SUID binaries",
            "Linux capabilities",
            "cron jobs",
            "application credentials",
            "local services / sockets",
            "writable privileged files",
        ]

    attempted = [a.signature for a in state.recorded_actions[-20:]]

    lines = [
        "CTF INVESTIGATION SUMMARY",
        "OBJECTIVE: Find and verify the flag. Do not submit unless asked.",
        f"CATEGORY: {state.category.value}",
        f"TARGET: {', '.join(state.target_urls) or '(none)'}",
        f"CURRENT PHASE: {state.phase.value}",
        f"CURRENT OBJECTIVE: {state.current_objective}",
        f"RETRY COUNT: {state.retry_count}",
        "",
        "KNOWN FACTS:",
        *[_bullet(item) for item in (state.known_facts or ["(none)"])],
        "",
        "SUCCESSFUL TECHNIQUES:",
        *[_bullet(item) for item in (state.successful_actions[-12:] or ["(none)"])],
        "",
        "FAILED APPROACHES:",
        *[_bullet(item) for item in (state.failed_actions[-12:] or ["(none)"])],
        "",
        "ALREADY ATTEMPTED:",
        *[_bullet(item) for item in (attempted or ["(none)"])],
        "",
        "CONFIRMED HYPOTHESES:",
        *[_bullet(item) for item in confirmed],
        "",
        "REJECTED HYPOTHESES:",
        *[_bullet(item) for item in rejected],
        "",
        "UNEXPLORED:",
        *[_bullet(item) for item in (unexplored or ["(none)"])],
        "",
        f"PRIVILEGES: {state.privileges or '{}'}",
        f"EXECUTION: {state.execution_context or '{}'}",
        f"CANDIDATE FLAGS: {', '.join(state.candidate_flags) or '(none)'}",
        f"CONFIRMED FLAG: {state.confirmed_flag or '(none)'}",
        "",
        "Do not repeat completed actions.",
        "Do not inspect unrelated challenge files unless new evidence makes them relevant.",
        "Take the single highest-value unexplored action.",
    ]
    if state.hint:
        lines.insert(6, f"HINT: {state.hint}")
    if not should_inspect_distfiles(state):
        lines.append(
            "Do NOT inspect /challenge/distfiles; the described attack surface is remote."
        )
    if state.phase == InvestigationPhase.STUCK:
        lines.extend(["", STALL_MESSAGE])
    return "\n".join(lines)


def _bullet(item: str) -> str:
    return f"- {item}"


def orchestration_system_prompt(state: InvestigationState) -> str:
    priorities = {
        ChallengeCategory.WEB: (
            "Prioritize the target URL, HTTP recon, endpoints, forms, uploads, "
            "and server-side execution. Then privilege escalation if needed."
        ),
        ChallengeCategory.REVERSING: (
            "Prioritize file identification, strings, symbols, disassembly, and execution."
        ),
        ChallengeCategory.FORENSICS: (
            "Prioritize evidence identification, metadata, strings, and extraction."
        ),
        ChallengeCategory.CRYPTO: (
            "Prioritize cipher identification, encodings, keys, and known plaintext."
        ),
        ChallengeCategory.PWN: (
            "Prioritize binary properties, protections, input surface, and crash analysis."
        ),
    }
    priority = priorities.get(
        state.category,
        "Choose tools from evidence. Prefer high-information actions.",
    )
    dist = (
        "Local distfiles may be relevant."
        if should_inspect_distfiles(state)
        else (
            "Do not inspect /challenge/distfiles first. Start with the described "
            "remote target. Only use local files if evidence later requires it."
        )
    )
    return "\n".join(
        [
            "You are an autonomous CTF solver.",
            "Your objective is to obtain and verify the flag.",
            "You are operating inside an authorized CTF environment.",
            "Be evidence-driven.",
            "Do not repeat actions that have already been attempted unless new "
            "evidence, a privilege change, or verification requires it.",
            "Prioritize the challenge's described attack surface.",
            "Maintain awareness of: current phase, current objective, privileges, "
            "known facts, failed approaches, and successful techniques.",
            "Prefer high-information actions.",
            "When an exploit primitive is confirmed, build on it.",
            "When an approach fails repeatedly, change strategy.",
            "When a flag is found, stop. Do not continue reconnaissance.",
            "Do not submit the flag unless the submit_flag tool is required for verification.",
            "Return the verified flag to the operator.",
            f"Challenge category: {state.category.value}. {priority}",
            dist,
        ]
    )


def state_from_challenge(
    description: str,
    hint: str = "",
    evidence_paths: list[str] | None = None,
    metadata_category: str = "",
    challenge_name: str = "",
    connection_info: str = "",
) -> InvestigationState:
    evidence_paths = list(evidence_paths or [])
    urls = extract_urls(description) + extract_urls(hint) + extract_urls(connection_info)
    category = classify_challenge(
        description,
        hint,
        evidence_paths,
        metadata_category,
        challenge_name,
    )
    state = InvestigationState(
        challenge_description=description,
        hint=hint,
        evidence_paths=evidence_paths,
        target_urls=urls,
        category=category,
    )
    if urls:
        state.services_of_interest.extend(urls)
        state.add_fact(f"Remote target(s): {', '.join(urls)}")
        if category == ChallengeCategory.WEB:
            state.current_objective = "Reach the web target and map the attack surface."
            state.next_best_action = f"Fetch {urls[0]}"
    elif should_inspect_distfiles(state):
        state.next_best_action = "Inspect challenge distfiles / evidence."
    return state
