## Summary

<!-- What does this change, and why? Link the issue it closes, if any. -->

## Testing

<!-- How was this verified? New tests, the existing suite, a numerical
     comparison against another solver (ECOS, SCS, Mosek)? -->

## Checklist

- [ ] The test suite passes locally (`julia --project -e 'using Pkg; Pkg.test()'`)
- [ ] New behavior is covered by tests in `test/`
- [ ] Changes to the solver interface are reflected in the MOI wrapper
- [ ] Docstrings and documentation are updated where relevant

## AI assistance

<!-- Check exactly one. Either way, you have personally reviewed and
     tested everything in this PR. -->

- [ ] No AI tools were used
- [ ] AI-assisted (suggestions, autocomplete, drafting) — commits carry
      `Assisted-by:` trailers
- [ ] Substantially AI-generated — commits carry `Generated-by:` trailers
