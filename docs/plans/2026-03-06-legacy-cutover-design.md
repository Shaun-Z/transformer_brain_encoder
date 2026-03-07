# Legacy Cutover Design

**Goal:** Remove the old `models/`, `datasets/`, `engine.py`, and related script-era training path only after all visible user-facing entrypoints have been moved to the new `src/` pipeline.

## Current State

The repository runtime is already centered on the rewritten `src/` stack, but the README and notebooks still reference:

- `python main.py ...` with legacy CLI flags
- `datasets.nsd` and `datasets.nsd_utils`
- `engine.py`
- old `brain_encoder_wrapper` constructor conventions
- machine-specific batch submission snippets

The remaining legacy code is no longer authoritative, but deleting it immediately would leave broken notebooks and stale documentation in the repository.

## Target State

The repo should have one authoritative workflow:

- training via `python -m src.cli.train`
- evaluation via `python -m src.cli.evaluate`
- artifact inspection via `python -m src.cli.inspect_data`
- prediction/export via `python -m src.cli.predict`
- inference from saved `.pth` artifacts via `brain_encoder_wrapper`

The notebooks should become thin clients that either:

- launch the new CLI, or
- read saved artifacts and visualize them

They should not import legacy training modules or depend on machine-specific paths.

## Migration Strategy

1. Rewrite the README to document the new workflow and artifact layout.
2. Update the notebooks to consume the new CLI and wrapper path.
3. Add regression coverage that ensures no code or docs still point to the legacy modules.
4. Delete `models/`, `datasets/`, `engine.py`, and other unused legacy training files.

## Verification

Cutover is complete only when:

- README examples use the new CLI
- notebooks no longer import legacy runtime modules
- tests still pass
- repo-wide search finds no active runtime references to `datasets.nsd`, `datasets.nsd_utils`, `engine.py`, or legacy `models.*` imports outside archival contexts
