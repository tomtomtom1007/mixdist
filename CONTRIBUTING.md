# Contributing to mixdist

## Development setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

The `dev` extra pulls in `gower`, which powers the cross-validation tests in
`tests/test_reference_gower.py`. Those tests skip when it is absent, so they are the
first thing to check if a distance change looks fine locally but is actually wrong.

## Checks

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest --doctest-modules src/mixdist -q
```

CI runs all three on Ubuntu × Python 3.9–3.13, macOS and Windows, plus a
minimum-dependency job (numpy 1.22 / pandas 1.4, no scikit-learn). Keep the library
importable and useful without scikit-learn — it is an optional dependency, not a hidden
one.

## Benchmarks

The numbers quoted in the README are produced by, and must stay consistent with:

```bash
.venv/bin/python examples/weighting_matters.py
.venv/bin/python examples/scaling.py
```

If a change moves them, update the README in the same commit. Claims in the README
should be reproducible by a reader running these scripts.

## Releasing

Publishing runs through [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
— there is no API token stored in the repository or in anyone's shell. PyPI verifies a
short-lived OIDC identity minted by GitHub for this exact repository, workflow file
(`publish.yml`) and environment (`pypi`).

To cut a release:

Nothing is published by pushing. `git push` runs CI; a bare tag does nothing. **Only
publishing a GitHub release triggers an upload.**

1. Bump `__version__` in `src/mixdist/__init__.py`. That is the *only* place a version is
   written — `pyproject.toml` reads it via `[tool.hatch.version]`, so the two cannot
   drift apart.
2. Move the `## [Unreleased]` entries in `CHANGELOG.md` under the new version with a date.
3. Commit, then tag and push:

   ```bash
   git tag -a vX.Y.Z -m "mixdist X.Y.Z" && git push origin main vX.Y.Z
   ```

4. Publish a GitHub release for the tag. That triggers `publish.yml`, which builds, runs
   `twine check --strict`, verifies the tag matches the built distributions, and uploads.

The tag check exists because **PyPI never accepts a re-upload of a version that already
exists**. Catching a forgotten bump in CI is cheap; burning a version number is not.

To exercise the build path without publishing, run the `Publish` workflow manually with
`dry_run` left at `true` — it builds and checks, then skips the upload job.

## Scope

The package deliberately stays narrow: distances, neighbour search and clustering for
mixed-type tables. Visualisation, AutoML and feature engineering belong elsewhere.
