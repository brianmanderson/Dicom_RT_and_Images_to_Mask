# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**DicomRTTool** — a published PyPI library (GPL-3.0, [cited paper](https://doi.org/10.1016/j.prro.2021.02.003), external users) that converts DICOM images + RT structures/doses into NIfTI files, NumPy arrays, and SimpleITK handles, and converts prediction masks back into RT structures.

- `src/DicomRTTool/ReaderWriter.py` — the main API (`DicomReaderWriter`, `ROIAssociationClass`); still a large class being incrementally extracted into `_internal/` (anonymizer, indexer, rt_contours). `Services/` holds DICOM base classes and the resampling helpers; the package ships two data files (`template_RS.dcm`, `key_list.txt`).
- `tests/` — **hermetic** suite: every DICOM file is synthesized in tmp at session start (`tests/synthetic.py`); no external corpus, no network. Keep it that way — never commit DICOM data.
- `evaluation/` — opt-in cross-tool harness vs the companion C# `DicomRtNifti.Cli` (needs external LCTSC data + built exe; see `evaluation/README.md`).

## Commands

Repo venv at `.venv/` (Python 3.12). CI (`.github/workflows/test.yml`) runs ruff + pytest on ubuntu/windows × Python 3.10–3.13.

```bash
pip install -e ".[dev]"                 # install (setuptools-scm needs git tags present)
.venv/Scripts/python.exe -m pytest -q   # full hermetic suite (~2 min on this machine)
.venv/Scripts/ruff.exe check .          # lint — CI gates on this, run before pushing
```

Three test groups auto-skip unless enabled:

- `tests/test_conformance.py` — analytical accuracy gate; needs `pip install -r requirements-conformance.txt` (a git-URL dep deliberately kept out of `pyproject.toml` — PyPI rejects direct-URL metadata). Separate CI check (`conformance.yml`).
- `tests/test_csharp_parity.py` — needs env vars `DICOMRTTOOL_LCTSC_DIR` and `DICOMRTTOOL_CSHARP_EXE`.
- `tests/test_real_corpus_dose.py` — dose fidelity against one real patient (CT + RTSTRUCT + RTDOSE); needs env var `DICOMRTTOOL_DOSE_CORPUS`.

## Versioning & release

- **Version comes from git tags via setuptools-scm** — there is no version string to edit. Between releases it reads `X.Y.Z.devN+g<sha>`.
- **Pushing a `v*` tag publishes to PyPI** (Trusted Publishing, `python-publish.yml`) — never tag casually.
- `CHANGELOG.md` is Keep-a-Changelog: put changes under `[Unreleased]`; a release commit promotes the section.

## Backwards compatibility

This library has external users. The public surface is `DicomRTTool/__init__.py`'s `__all__` plus the documented `DicomReaderWriter` attributes/methods (README shows the contract). Don't rename or remove public names without a deprecation path — v4 removed v3 names only after a documented cycle (`tests/test_compat.py` guards the current surface). Output formats are contracts too: `metadata.json` is schema-versioned, and anonymization hashes must stay byte-identical to the companion C# tool.

## Pitfalls

- **Check git state before starting work.** Local `main` and the checked-out feature branch have been observed behind `origin/main` (which already carried newer release tags) — `git fetch` and confirm where `origin/main` and the latest `v*` tag sit first.
- `temptest.py` at the root is an untracked scratch script pointing at real patient data paths. Don't commit it, don't use it as an example, don't run it.
- Mask differences vs the C# tool are **by design**: DicomRTTool rasterizes boundary-inclusive, the C# tool fills the polygon interior. Dice ≈ 0.99 on large OARs is the expected agreement, not a bug (see `evaluation/README.md`).
- When searching, exclude `.venv/`, `__pycache__/`, `dist/`, `src/DicomRTTool.egg-info/`, and `.claude/worktrees/`.
- Requested extra DICOM tags are written verbatim into `metadata.json` — anonymized exports can still leak PHI through them.

## Further reading

- [README.md](README.md) — the full user-facing API tour (discover → survey → select → export), anonymization, resampling, metadata schema.
- [CHANGELOG.md](CHANGELOG.md) — per-release history and migration notes (v4 renames, v5+ export features).
- [evaluation/README.md](evaluation/README.md) — cross-tool comparison method and expected tolerances.
