
# CTF Agent

CTF Agent is a Python-based assistant for solving authorized Capture The Flag
(CTF) challenges. It uses a single LLM-driven solver with a Docker workspace
containing common tools for web, crypto, reversing, pwn, forensics, and
steganography challenges.

Use it only against challenges, systems, and files you are explicitly
authorized to test.

## How it works

```text
CTF challenge → Solver → LLM tool calls → Docker sandbox → candidate flag → CTFd
```

For each run, the solver:

- reads challenge metadata and attached files;
- classifies the challenge category;
- runs shell, file, image, and HTTP tools in the sandbox;
- tracks its investigation state, repeated actions, requests, and model cost;
- submits a candidate flag to CTFd, unless `--no-submit` is used.

Challenge files are mounted read-only at `/challenge/distfiles`; generated
artifacts belong in `/challenge/workspace`.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- Docker Engine
- Credentials for one configured model provider
- Optional: access to a CTFd instance for downloading challenges and submitting flags

## Setup

Clone the repository and install Python dependencies:

```bash
git clone https://github.com/AungMoeYan/my-ctf-agent.git
cd my-ctf-agent
uv sync
```

Build the sandbox image:

```bash
docker build -f sandbox/Dockerfile.sandbox -t ctf-sandbox .
```

Create your local environment file:

```bash
cp .env.example .env
```

At minimum, configure the CTFd address and token when you intend to submit
flags. Do not commit `.env` or provider credentials.

```dotenv
CTFD_URL=https://ctf.example.com
CTFD_TOKEN=ctfd_your_api_token_here

# Bedrock example
AWS_REGION=us-east-1
AWS_BEARER_TOKEN=your_bedrock_token
```

## Usage

Run a downloaded challenge directory:

```bash
uv run ctf-solve --challenge challenges/example --no-submit
```

Provide a challenge directly in the terminal:

```bash
uv run ctf-solve \
  --description "Recover the flag from the attached binary." \
  --evidence ./challenge-files \
  --no-submit
```

Run interactively by omitting `--challenge`, `--description`, and
`--instruction`:

```bash
uv run ctf-solve --no-submit
```

Useful options:

| Option | Purpose |
| --- | --- |
| `--model provider/model` | Select a model; the default is `bedrock/deepseek.v3.2`. |
| `--image IMAGE` | Use a different Docker sandbox image. |
| `--ctfd-url URL` / `--ctfd-token TOKEN` | Override the values in `.env`. |
| `--hint TEXT` | Supply a challenge hint. |
| `--evidence PATH` | Add a local file or directory as evidence; repeat as needed. |
| `--no-submit` | Treat candidates as dry-run results; use this first. |
| `-v` | Enable verbose logs. |

## Downloading challenges from CTFd

Use the standalone pull script to fetch available challenges and their
attachments:

```bash
uv run python pull_challenges.py \
  --url https://ctf.example.com \
  --token ctfd_your_api_token_here \
  --output challenges
```

Username/password login is also supported:

```bash
uv run python pull_challenges.py \
  --url https://ctf.example.com \
  --username your-user \
  --password your-password
```

## Model providers

Model specifications use `provider/model-id` format. The current solver
supports `bedrock`, `azure`, `zen`, and `google` providers. Configure the
corresponding environment variables in `.env`; see `backend/config.py` and
`backend/models.py` for the exact names.

## Architecture and development status

CTF Agent combines cloud-hosted LLM providers with containerization:

- **Cloud providers:** supports Bedrock, Azure OpenAI, OpenCode Zen, and
  Google AI models through Pydantic AI.
- **Containerized tools:** each solver runs analysis tools in a Docker sandbox
  with a read-only challenge mount and a separate writable workspace.
- **CTFd integration:** can download challenge data and submit confirmed flags
  to a CTFd instance.

> [!WARNING]
> This project is under active development. Use it only for authorized CTF
> challenges and begin with `--no-submit`.

### Development process

The current development priorities are:

1. Pass all operator instructions, hints, and evidence into the solver prompt.
2. Add cooperative cancellation between LLM requests and tool calls.
3. Keep the architecture consistent: complete multi-agent coordination or
   remove its remaining hooks and remain single-agent.
4. Separate LLM-request budgets from tool-call budgets.
5. Add end-to-end tests for flag submission, dry runs, cancellation, prompt
   context, and sandbox behavior.
6. Treat challenge content and tool output as untrusted data to reduce
   prompt-injection risk.

## Development

Run the test suite:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

## Security notes

- Start with `--no-submit`; it prevents an actual CTFd submission.
- The sandbox has CPU and memory limits and does not use the previous
  privileged Docker capabilities. It still has network access for legitimate
  CTF targets, so do not use it with unknown or untrusted target scope.
- Keep API keys and CTFd tokens in `.env` only.
