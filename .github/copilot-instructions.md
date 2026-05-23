# Copilot / Agent Instructions (bootstrap)

Purpose
-------
Provide a short, practical guide for AI assistants (Copilot/chat agents) to work productively in this repository.

Quick summary
-------------
- Repo root contains Python utilities and data files. Primary entrypoint: `main.py`.
- This document is intentionally brief — link to docs rather than duplicate them.

Workflow for the agent
----------------------
1. Discover conventions: look for `.github/copilot-instructions.md`, `AGENTS.md`, `AGENT.md`, `README.md`, and `docs/`.
2. Explore: identify build/test commands, architecture, and areas with special conventions.
3. Propose or apply small changes (PRs) that follow existing style and test locally when possible.
4. Ask clarifying questions if assumptions are required.

Repository specifics
-------------------
- Language: Python. Entrypoint: `main.py`.
- Virtualenv: `venv/` present in the workspace; use the project's preferred environment if available.

Apply-to and scope guidance
--------------------------
- Global agent instructions live here (`.github/copilot-instructions.md`).
- If you create area-specific instructions later, use `applyTo` patterns (e.g., `path:frontend/**`) and keep them small.

Example prompts
---------------
- "Add a unit test for the function that loads `user_profiles_improved.json`."
- "Create `.github/workflows/test.yml` to run `python -m pytest` on pushes and PRs."

Next customizations to consider
------------------------------
1. `AGENTS.md` focused on repo-level agent roles and expectations.
2. Small `copilot-instructions` per major folder (if project grows).

How to give feedback
--------------------
If this document is missing needed context, ask the maintainers for the preferred local workflow (env setup, test commands, linters), then update this file and link to authoritative docs rather than embedding long instructions.

--
This file was added as a concise bootstrap; maintainers should expand or split sections as the repo grows.
