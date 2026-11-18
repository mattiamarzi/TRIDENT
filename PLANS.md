# Version 0.2.0 publication preparation

Objective: restore the already prepared 0.2.0 changes in the user-selected
directory, verify source/tests/build artifacts, then provide manual upload
instructions. Publishing the prepared source to GitHub is authorized; publishing
to PyPI remains a separate step.

Invariants: preserve models, solver settings, node order, motif counts,
vectorized kernels, public output structures, and unrelated user files.

Affected files: validator modules, legacy BiPLOM projection, version metadata,
concise documentation, regression tests, project log, and ignored build outputs.

Stages: compare the input tree to reviewed PR #1; back up affected files; restore
the verified changes; run all tests; build and check sdist/wheel; verify the
final wheel in a separate environment and inspect differences.

Risks: approximation results may differ from 0.1.0. Full HCP analysis and
cross-platform CI are outside this check. Local tests do not establish a passing
remote CI result. This directory has no Git metadata.

Completion: 96 tests pass against source and rebuilt wheel, artifact checks
pass, version is consistently 0.2.0, and exact upload instructions are supplied.

Progress: input files matched the reviewed PR exactly. Existing files backed
up before restoring prepared changes. All 96 tests passed against both source
and the rebuilt wheel installed in a separate environment. The sdist and wheel
passed twine check. Source publication to GitHub is the final remaining step.
