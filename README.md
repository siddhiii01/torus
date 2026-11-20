# Torus

Local document engine: any PDF or image → fill the columns of a client Excel template.

Confirmed values need independent agreeing signals; uncertain values go to human review. Quality target: zero silent errors.

## Install

See [docs/SETUP.md](docs/SETUP.md) for environment variables, hardware notes, and optional model stacks.

```bash
git checkout dev-sid
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras (install only what the machine can run):

```bash
pip install -e ".[paddle]"
pip install -e ".[doctr]"
pip install -e ".[vlm]"
pip install -e ".[train]"
pip install -e ".[review]"
```

## Agent rules

Read [AGENTS.md](AGENTS.md) before changing code. Work is picked from [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).
