# Project log

## 2026-11-18 — Prepare version 0.2.0

- Objective: publish the prepared source update and provide packages for updating
  the existing trident-validator project on PyPI.
- Baseline: merged PR #1, main commit 0a62162. The supplied source archive matched
  the reviewed PR files exactly. Existing affected files were backed up locally
  before restoring the prepared changes.
- Updated pair/triplet normal and refined-normal approximations consistently
  with the documented inclusive upper tail. Preserved models, solver settings,
  motif counts, node ordering, output structures and vectorized matrix kernels.
  Approximation p-values and validated sets can differ from version 0.1.0.
- Fixed adjacent legacy BiPLOM projection defects: lower-tail output instead of
  upper-tail output, zero-count and deterministic cases, and a missing PoiBin
  import. The exact distribution implementation itself was not changed.
- Version 0.2.0 in pyproject.toml, package __version__ and CITATION.cff. Removed
  the stale citation release date pending publication. Updated README and math/API
  documentation without repeating the existing vectorization description.
- Added tests/test_inclusive_tail.py with 84 analytic and regression cases.
  Independent symmetric-binomial references fail against the original formula;
  skewed references, deterministic cases, both validators, all approximation
  engines, and legacy/public method parity are covered.
- Final source suite: 96 passed (Python 3.10.13, NumPy 1.24.0, SciPy 1.15.3,
  pandas 2.2.3, pytest 7.4.4). Three existing invalid-escape docstring warnings.
- Final rebuilt wheel installed in a separate environment: 96 passed
  (NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, pytest 9.2.0, no Numba).
- Source and wheel validation command: python -m pytest -q. Build command:
  python -m build --no-isolation. Both source distribution and wheel passed
  python -m twine check. Build reports pre-existing deprecated license metadata.
- Earlier validation of the same prepared numerical code on the supplied
  116 x 2396 dataset with fitted BiCM found identical validated sets between
  engines for all three approximation methods and global/Bonferroni/FDR rules;
  largest absolute p-value difference 5.774e-15. It was not a comparison against
  the old scientific results, and no empirical equivalence to 0.1.0 is claimed.
- No dependency or LaTeX package was added; no benchmark improvement is claimed.
- Limitations: all 199 HCP scans, full empirical BiPLOM refitting, manuscript,
  notebook execution, Python 3.11/3.12 and Linux/Windows remain unverified.
  No PyPI upload was performed as part of source preparation.
