# Contributing to ConicIP.jl

Thanks for your interest in contributing! Bug reports, fixes, and
documentation improvements are all welcome. Please run the test suite
(`julia --project -e 'using Pkg; Pkg.test()'`) before opening a pull
request, and fill in the PR template.

## AI-assisted contributions

AI coding tools are welcome in this project. We ask only that their
involvement is visible.

If a tool meaningfully assisted a commit, disclose it with a git
trailer:

    Assisted-by: Claude Code

If a contribution is substantially machine-generated, use instead:

    Generated-by: Claude Code

Substitute the name of whatever tool you used. Do not use
`Co-authored-by:` for AI tools, and never attach a `Signed-off-by:`
line on behalf of a tool — sign-off means *you* vouch for the change.

The pull request template asks you to check one AI-assistance box.
Answer it honestly; it is there to help reviewers calibrate their
attention, not to police tool use. All contributions, assisted or
not, must be reviewed and tested by the human submitting them.
