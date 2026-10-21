# ConicIP.jl

The project guide (architecture, tests, docs, observability, environment)
is in AGENTS.md, shared with other coding agents:

@AGENTS.md

## Commit and PR conventions

- Never add `Co-Authored-By` lines to commits.
- When you author a commit, append the trailer:

      Assisted-by: Claude Code

  If the commit is substantially machine-generated, use instead:

      Generated-by: Claude Code

- When you open a PR or issue, or write a substantive comment,
  end with one line: "Drafted with Claude Code."
- Never add a `Signed-off-by` line. Sign-off is reserved for humans.
- Do not add any disclosure to trivial edits dictated verbatim by
  the maintainer.
- When opening a PR, fill in the template at
  `.github/pull_request_template.md`, including the AI-assistance
  checkbox that matches the commits.
