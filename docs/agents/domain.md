# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — domain glossary for DSA-110 continuum imaging. Each entry that names a code-level fact carries a `<path>::<Symbol>` citation; if you need ground truth, follow the citation.
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.
- **`docs/reference/`** — distilled deep references (calibration parameters, WSClean flags, validated failure modes) for the area you're touching. Read alongside `CONTEXT.md`, not as a substitute.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo:

```
/
├── CONTEXT.md           ← glossary
├── docs/
│   ├── adr/             ← architectural decisions
│   ├── reference/       ← deep references (existing)
│   └── skills/          ← verified code-behaviour notes (existing)
└── dsa110_continuum/    ← package
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

Examples:
- Say *tile*, not "snapshot" or "frame".
- Say *hourly-epoch mosaic*, not "daily mosaic".
- Say *Dec strip*, not "RA strip" — the persistent geometric concept is constant declination; RA-strip framing is a legacy artifact of the day-batch partition.

If the concept you need isn't in the glossary, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Verifying citations

`CONTEXT.md` entries cite `path/to/file.py::Symbol`. To check that the cited symbols still resolve, run:

```
/opt/miniforge/envs/casa6/bin/python scripts/verify_glossary.py
```

This parses the glossary, opens each cited Python file, and verifies the symbol exists at the expected location. Run after any refactor that renames functions, classes, or modules.

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (per-date gain solutions) — but worth reopening because…_
