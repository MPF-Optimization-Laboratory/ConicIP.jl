#!/bin/zsh
# Create release-<ver> from dev with the private development artifacts removed.
# Usage: benchmark/strip-release.sh 0.6.0   (run from any worktree of the repo)
set -e
ver=${1:?version, e.g. 0.6.0}
root=$(git rev-parse --show-toplevel)
repo=$(git -C "$root" rev-parse --git-common-dir)/..
wt="$repo/.worktrees/release-$ver"
git -C "$repo" worktree add -b "release-$ver" "$wt" dev
cd "$wt"
git rm -q \
  benchmark/suite.jl benchmark/baselines.jl benchmark/moi_model.jl benchmark/compare.jl \
  benchmark/sumnorms_diag.jl benchmark/Project.toml benchmark/large-scale-roadmap.md \
  benchmark/phases.jl benchmark/compare_phases.jl benchmark/alloc_phases.jl benchmark/setup_diag.jl \
  benchmark/hsd-design.md benchmark/RESUME.md benchmark/strip-release.sh \
  test/review_benchmark.jl test/harness_tests.jl
echo "Removed the private paths on branch release-$ver in $wt."
echo "Remaining references to clean by hand:"
grep -rn -e "suite.jl" -e "large-scale-roadmap" -e "hsd-design" -e "baselines" -e "moi_model" \
        -e "sumnorms" -e "phases.jl" -e "alloc_phases" -e "setup_diag" -e "review_benchmark" -e "harness_tests" -e "benchmark/results" \
        -e "benchmark/.cache" -e "benchmark/Manifest" -e "RESUME.md" -e "strip-release" \
        CLAUDE.md CHANGELOG.md README.md .gitignore test docs/src src benchmark 2>/dev/null || true
echo "Then: remove Dates from test/Project.toml if unused, bump Project.toml, add the CHANGELOG header, test, build docs, push, PR."
