# Shared Codex context

Root `AGENTS.md` is the entry point for Codex instructions. Start an existing Codex
client in this repository (CLI: `cd /path/to/Tdyssey` then `codex`), or select this
folder in the app/IDE. No Codex installation is performed by this scaffold.

Codex discovers project instructions when a session starts. After pulling changed
guidance, start a fresh session or explicitly ask the active agent to reread it.
Global instructions and deeper `AGENTS.override.md` files can change the effective
instructions; do not assume all collaborators have identical personal settings.
The shared `.codex/config.toml` contains only workspace/approval preferences. Its
loading depends on client support and project trust; each author controls their own
trust setting. It does not enforce package versions or share private conversations.

Suggested first prompt:

```text
Read AGENTS.md, docs/STATUS.md, and the relevant project documents. Summarize the
current stage, verified evidence, and next milestone. Use the environment/profile
available for the assigned task. Respect the machine's current setup permissions
and storage budget.
```

Assign a concrete task and expected verification rather than asking an agent to
implement the entire schedule at once. After substantial work, the agent updates
status and decisions so another author's next session can continue from Git.
Keep secrets, personal account configuration, and absolute workstation paths out
of shared files. This setup has no automated install hooks or background jobs.

Small source-only changes use `python3 -B scripts/check_repo.py`. Runtime work uses
the locked profile only on an authorized machine and records what was actually
executed. Machine setup permissions are a working agreement, not an OS security
boundary; account/organization policies still govern execution.

Official references:

- [AGENTS.md discovery](https://developers.openai.com/codex/guides/agents-md)
- [Project config and trust](https://developers.openai.com/codex/config-basic)
