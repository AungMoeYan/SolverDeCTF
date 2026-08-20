# CTF Agent


An autonomous AI-powered Capture The Flag (CTF) solving agent designed to investigate challenges, use security tools, execute commands inside an isolated Docker sandbox, and return the discovered flag.


The agent currently uses **PydanticAI + AWS Bedrock + DeepSeek** as the reasoning layer.


> **Important:** The agent is designed to **find and return flags**. Automatic flag submission is intentionally outside the solver's responsibilities.


---


## Overview


The goal of this project is to build an autonomous CTF researcher that can take a challenge description and independently investigate it.


```text
                    CTF Challenge
                         │
                         ▼
                ┌─────────────────┐
                │    Coordinator  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     Solver      │
                │   PydanticAI    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   AWS Bedrock   │
                │    DeepSeek     │
                └────────┬────────┘
                         │
                    Tool Calls
                         │
                         ▼
                ┌─────────────────┐
                │ Docker Sandbox  │
                │  CTF Tools      │
                └────────┬────────┘
                         │
                         ▼
                  Tool Results
                         │
                         └──────────► Solver
                                      │
                                      ▼
                                Flag Detection
                                      │
                                      ▼
                                 Return Flag

The solver follows an iterative loop:

Understand
    ↓
Plan
    ↓
Execute
    ↓
Observe
    ↓
Update
    ↓
Repeat
    ↓
Flag Found
    ↓
Stop
Architecture
Main Components
Coordinator

Responsible for high-level challenge execution and coordinating solver runs.

Solver

The main autonomous reasoning component.

Location:

backend/agents/solver.py

The solver:

receives the challenge description
reasons about the challenge
chooses tools
executes actions
analyzes results
continues investigation
detects successful exploitation
returns the flag

The solver is the primary component currently being improved.

Model Layer

The project uses PydanticAI for model interaction.

Current primary model:

AWS Bedrock
└── DeepSeek V3.2

Model configuration and provider resolution are handled by the existing model infrastructure.

Docker Sandbox

CTF commands are executed inside an isolated Docker environment.

The sandbox provides:

Linux environment
CTF utilities
filesystem isolation
command execution
network access where required by challenges

The solver should interact with the challenge through the existing sandbox/tool infrastructure rather than directly modifying the host system.

Tools

Tools provide the model with capabilities such as:

shell commands
web requests
file inspection
filesystem operations
reverse engineering utilities
challenge-specific investigation capabilities

The solver decides which tools are useful based on the challenge.

Repository Structure

The exact repository structure is authoritative. The following describes the important components:

ctf-agent/
│
├── backend/
│   ├── agents/
│   │   └── solver.py
│   │
│   ├── tools/
│   │   └── ...
│   │
│   ├── deps.py
│   ├── models.py
│   └── ...
│
├── sandbox/
│   └── ...
│
├── tests/
│   └── ...
│
├── test-challenges/
│   └── ...
│
├── pyproject.toml
├── README.md
└── ...

When making changes, inspect the actual source tree rather than assuming this documentation is more authoritative than the implementation.

Solver

The main solver is:

backend/agents/solver.py

The solver uses PydanticAI to create an interactive agent capable of repeatedly calling tools.

Conceptually:

Challenge
    │
    ▼
Create Solver Context
    │
    ▼
LLM Request
    │
    ▼
Model Chooses Action
    │
    ▼
Tool Execution
    │
    ▼
Tool Result
    │
    ▼
LLM Receives Result
    │
    ├── Need more investigation ──► Tool
    │
    └── Flag found ───────────────► Finish

The solver is intentionally iterative because many CTF challenges require multiple dependent actions.

For example:

Discover endpoint
      ↓
Identify vulnerability
      ↓
Exploit vulnerability
      ↓
Obtain access
      ↓
Escalate / pivot
      ↓
Locate flag
Current Model

The current production/test configuration uses:

Provider: AWS Bedrock
Model: DeepSeek V3.2

Example model identifier:

bedrock/deepseek.v3.2

AWS credentials and other secrets must never be committed to this repository.

Configuration should be supplied through the existing configuration/environment mechanism.

Sandbox

The sandbox isolates challenge execution from the host environment.

Typical challenge paths include:

/challenge/
├── distfiles/
└── workspace/

The distinction between environments is important:

Host
 │
 └── CTF Agent
       │
       ▼
   Docker Sandbox
       │
       ├── /challenge/distfiles
       └── /challenge/workspace
       │
       ▼
   Remote CTF Target

The solver may need to interact with both:

Local challenge artifacts inside the sandbox.
Remote challenge infrastructure supplied by the CTF.

The existence of local files does not automatically mean they are relevant to the challenge.

Challenge Classification

A strong solver should first identify the likely challenge category.

Supported conceptual categories include:

Web
Reverse Engineering
Pwn
Crypto
Forensics
OSINT
Steganography
Misc
Programming

Classification should influence tool selection.

For example:

Web
HTTP
URLs
curl
web requests
source inspection
file uploads
authentication
sessions
web application behavior
Reverse Engineering
file
strings
readelf
objdump
gdb
static analysis
dynamic analysis
binary execution
Forensics
file
strings
metadata
binwalk
archives
disk analysis
packet analysis

The solver should avoid investigating unrelated artifacts simply because they exist in the sandbox.

Investigation Strategy

The solver should be goal-oriented and evidence-driven.

The objective is not to maximize the number of tool calls.

The objective is:

Maximum useful information
──────────────────────────
      Minimum actions

Every action should ideally answer a question.

Bad:

Run random enumeration
↓
Run another enumeration
↓
Run another unrelated tool

Better:

Hypothesis:
The upload endpoint may allow executable files.


Action:
Upload a controlled PHP file.


Observation:
PHP executes.


Conclusion:
Unrestricted server-side code execution exists.


Next objective:
Determine the privileges and identify a path toward the flag.
Investigation State

The solver should maintain awareness of important discoveries throughout an investigation.

Important state includes:

Challenge
Target
Challenge Type
Current Objective
Current Hypothesis


Known Endpoints
Known Technologies
Successful Techniques
Failed Techniques
Executed Actions


Current Privilege
Important Files
Important Responses


Flag Candidates
Final Flag

The solver should not repeatedly rediscover information that has already been established.

Progress

An action should produce one of the following:

NEW INFORMATION
CONFIRMATION
FAILURE
NO PROGRESS

Repeated actions that produce no new information should cause the solver to reconsider its strategy.

For example:

Upload PHP → executes

is useful information.

Repeating:

Upload PHP → executes
Upload PHP → executes
Upload PHP → executes

without a new objective is not useful.

Request Efficiency

Model requests are limited and expensive.

The solver therefore needs to minimize unnecessary reasoning cycles.

A previous test exposed an important failure mode:

Useful investigation
        ↓
Repeated actions
        ↓
Request limit reached
        ↓
Compaction
        ↓
Investigation restarts
        ↓
Previously completed work repeated
        ↓
Request limit reached again

This is considered a solver orchestration problem.

Increasing the request limit alone is not considered a proper solution.

The preferred approach is:

better state tracking
better progress detection
action deduplication
better challenge classification
focused tool selection
compact investigation summaries
early termination
preservation of useful state across compaction
Flag Detection

The primary success condition is finding the flag.

Common formats include:

flag{...}
picoCTF{...}
CTF{...}

However, the solver should not rely solely on a regular expression.

A candidate should be evaluated using the surrounding evidence.

Once a highly credible flag is discovered:

STOP INVESTIGATION
        ↓
RETURN FLAG

The solver does not need to submit the flag automatically.

Error Handling

The solver should gracefully handle:

tool failures
command failures
network failures
sandbox failures
model failures
request-limit exhaustion
context compaction
unsuccessful investigations

Request-limit exhaustion should result in a controlled solver outcome rather than an unhandled exception.

Known Development Issues

The current solver is functional but still has orchestration inefficiencies.

The most important known issue is investigation repetition.

A previous web challenge demonstrated this behavior:

1. Discover website
2. Discover upload endpoint
3. Upload PHP
4. Confirm PHP execution
5. Attempt /root
6. Receive permission denied
7. Repeat upload testing
8. Re-run reconnaissance
9. Revisit unrelated local artifacts
10. Exhaust model request budget
11. Compact
12. Restart
13. Repeat previous investigation
14. Exhaust budget again

The desired behavior is:

Discover upload
      ↓
Confirm PHP execution
      ↓
Establish current privilege
      ↓
Analyze permission boundary
      ↓
Change strategy
      ↓
Find next useful path
      ↓
Find flag
      ↓
STOP
Development Principles

When modifying this project:

Preserve
existing sandbox
existing tools
existing CLI
PydanticAI integration
Bedrock integration
model configuration
working functionality
Prefer
small changes
clear state management
deterministic behavior where possible
useful logging
tests
graceful failures
efficient model usage
Avoid
rewriting the entire project
duplicating existing tools
hardcoding challenge solutions
hardcoding flags
blindly increasing request limits
creating unnecessary abstractions
replacing working infrastructure without justification
AI Coding Agent Guidance

This repository may be modified using AI coding agents such as:

Antigravity
Z.ai
Cursor
Codex

Before modifying code, an AI coding agent should:

Read this README.
Inspect the actual repository.
Inspect backend/agents/solver.py.
Inspect the tools used by the solver.
Inspect dependencies and model configuration.
Understand the existing execution flow.
Identify the smallest change required.
Preserve existing interfaces where possible.
Test changes before considering the task complete.

The README describes the intended architecture, but the source code is authoritative.

AI agents must not assume that a feature exists merely because it is described here.

Development Workflow

Recommended workflow:

Inspect
  ↓
Understand
  ↓
Identify Problem
  ↓
Design Minimal Change
  ↓
Implement
  ↓
Test
  ↓
Review

For solver changes, specifically verify:

model request count
tool-call count
duplicate actions
investigation state
retry behavior
compaction behavior
flag detection
termination behavior
error handling
Testing

The solver should eventually be tested against multiple CTF categories.

Important test cases include:

Simple filesystem flag
Web challenge
Web file upload
Reverse engineering
Forensics
Multi-step exploitation
Repeated failed actions
Tool failure
Request budget exhaustion
Context compaction
Early flag discovery

A successful solver should stop as soon as the flag is reliably identified.

Project Status

Current status:

Core agent             █████████░  Working
Bedrock integration    █████████░  Working
DeepSeek integration   █████████░  Working
Docker sandbox         █████████░  Working
Tool execution         █████████░  Working
Autonomous solving     ███████░░░  Improving
Investigation state    █████░░░░░  Improving
Request efficiency     █████░░░░░  Improving
Flag detection         ███████░░░  Improving

The primary development focus is currently improving the solver's reasoning efficiency, state management, and ability to avoid repetitive investigation.
