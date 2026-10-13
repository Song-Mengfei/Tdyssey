# Commits and pull requests

This repository uses [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
for commit and PR titles. This is our shared convention; GitHub does not prescribe
one universal message format. Write for another author who has not read the chat.

## Commit titles

```text
type(scope): short description
```

Scope is optional. Use lowercase type/scope, a clear action such as `add`, `fix`,
or `record`, and a title of at most 72 characters. Avoid vague titles such as
`updates`, `changes`, or `fix stuff`. Mark an incompatible change with `!` and
explain what callers must change in the body.

| Type | Use |
| --- | --- |
| `feat` | New behavior or capability |
| `fix` | Correct an error |
| `docs` | Documentation changes |
| `test` | Add or improve meaningful tests |
| `refactor` | Restructure code without changing behavior |
| `perf` | Improve measured performance |
| `build` | Dependencies or build configuration |
| `ci` | GitHub checks and automation |
| `chore` | Repository maintenance and shared working rules |
| `style` | Formatting only |
| `revert` | Undo a change; identify the original commit |

Suggested scopes: `env`, `data`, `wm`, `policy`, `eval`, `deps`, `git`, `codex`,
or `repo`. Use a more specific scope when it helps. For example:

```text
feat(env): add L-shaped object support
fix(eval): account for I-shape rotational symmetry
build(deps): pin compatible Zarr and NumCodecs versions
docs(project): record failed imagined-training experiment
```

## Commit information

For a substantive change, use this body after a blank line:

```text
Why:
Explain the problem or reason for the change.

Changes:
- Describe the resulting behavior or shared rule.

Validation:
- Exact check command and result.
- State any checks that could not run and why.

Refs: #actual-issue-or-PR-number
```

The reference is optional; omit it when no related issue/PR exists. A tiny spelling
fix can use only a clear title. Describe actual work, not abandoned attempts or
promised results. Never invent issue numbers, test results, signatures, or author
credit. Use the author's configured Git identity; confirm an available authenticated
identity if none is configured. Never borrow another author's identity.

The shared editor template is `.github/commit-message-template.txt`. Each author
may enable it locally from the repository root:

```bash
git config --local commit.template .github/commit-message-template.txt
```

This is a local Git setting, not a package installation. Git does not activate
templates for other authors automatically. Before a scripted commit, write its
exact message to a temporary file, check it, and pass it to `git commit -F`:

```bash
python3 -B scripts/check_messages.py --commit-message /path/to/message.txt
git commit -F /path/to/message.txt
```

## Pull request information

Use the same title format. Fill `.github/pull_request_template.md`:

- **Summary:** the problem and resulting behavior, with a before/after example if useful.
- **Changes:** the important changes, sized to the work.
- **Validation:** commands/results, actual experiments, or reasons checks did not run.
- **Limitations:** untested behavior, remaining uncertainty, or `None` if appropriate.
- **Related issues:** genuine references, or `None`.

Keep it concise. Do not leave template instructions or empty sections. Use `Closes`
only when the PR actually resolves that issue. Keep research results distinct from
plans and distinguish file checks from actual training/simulator verification.
Use `--body-file` for GitHub CLI descriptions so newlines stay intact.

GitHub checks validate PR title format and nonempty sections, not the truth of
claims. Local checks validate commit title format and body spacing. Body content
and research evidence still require author/reviewer judgment. The template becomes
the GitHub default once merged into the default branch.

## Shared Git workflow

Work on a separate branch/checkout; inspect and stage only the intended changes.
Keep commits focused on one purpose. Open a draft PR for unfinished work and
update its title/body to describe the final change when scope changes. Have another
coauthor review shared changes before merging. Prefer a squash merge using the
formatted PR title when the team authorizes merging; do not merge automatically.

Do not amend/force-push published shared history without explicit authorization.
Older commits predate this convention and do not need rewriting. No Git hooks,
reviewer assignments, or branch-protection changes are installed by these rules.

Reference: [GitHub PR templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository).
