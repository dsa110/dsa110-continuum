# Plan: Unify GitHub contributors-list attribution to a single user

## Goal

On `github.com/dsa110/dsa110-continuum`, collapse the contributors page to show only `jakobtfaber` (the repo owner). Currently, Cursor Agent, GitHub Copilot, and Claude (via Co-Authored-By trailers) each appear as separate contributors in addition to the owner. The owner wants all those agent-attributed contributions to appear as their own.

## Current state (verified locally)

- 235 commits on `main`.
- **4 raw-author bot commits** on main:
  - 3 commits with raw author `Cursor Agent <cursoragent@cursor.com>` (`0fd4c8c`, `34f1979`, `8ceed34`)
  - 1 commit with raw author `copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>` (`9538ffd`)
- **38 commits** on main contain `Co-Authored-By:` trailers naming agents:
  - Claude Opus 4.7 / Opus 4.6 / Opus 4.6 (1M context) / Sonnet 4.6 — all with email `noreply@anthropic.com` (~58 trailer occurrences across ~36 commits)
  - Cursor Agent <cursoragent@cursor.com> (1 trailer)
  - Copilot <175728472+Copilot@users.noreply.github.com> (2 trailers)
- 3 other local branches besides main: `codex/diagnose-ms-field-discovery`, `integration/safe-subset-cloud`, `wip/pre-align-main-20260430T053503Z`.
- A `.mailmap` already exists mapping `Cursor Agent <cursoragent@cursor.com>` → `Jakob Faber <37250147+jakobtfaber@users.noreply.github.com>` (display-only; doesn't affect github.com attribution).

## Mechanism understanding (with caveats from a recent fact-check)

- github.com's contributors graph attributes commits by raw author email, matched against verified emails on a GitHub account. `.mailmap` does NOT affect github.com attribution.
- `Co-Authored-By:` trailers in commit message bodies CAN add a co-author to the contributors graph if the trailer email is verified on a GitHub account. They show as small avatars on PR/commit pages regardless.
- There is empirical filtering on the contributors API endpoint (e.g. Cursor Agent's 3 raw-author commits on main don't appear in the API response despite the `cursoragent` user existing publicly), but the contributors *page* surface may include them. We're optimizing for the page surface.

## Proposed operation

### Step 1 — Install `git-filter-repo` (single-file Python script)

`git-filter-repo` is the modern, supported tool for history rewrites (replaces deprecated `git filter-branch`). The single-file form is one Python script — no system package install, no global pip pollution. Drop into `~/.local/bin/git-filter-repo` and chmod +x.

### Step 2 — Backup before rewrite

Push backup branches to origin:

```
git push origin main:refs/heads/main-pre-agent-unify-2026-05-05
git push origin codex/diagnose-ms-field-discovery:refs/heads/codex-diagnose-pre-agent-unify-2026-05-05
git push origin integration/safe-subset-cloud:refs/heads/integration-safe-subset-pre-agent-unify-2026-05-05
git push origin wip/pre-align-main-20260430T053503Z:refs/heads/wip-pre-align-pre-agent-unify-2026-05-05
```

Reflog also preserves originals locally for ~90 days.

### Step 3 — Extend `.mailmap`

Add the Copilot bot mapping alongside the existing Cursor Agent one:

```
Jakob Faber <jfaber@caltech.edu> Cursor Agent <cursoragent@cursor.com>
Jakob Faber <jfaber@caltech.edu> copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>
```

Note: the rewritten author email is `jfaber@caltech.edu` (consistent with the 5 commits already reauthored earlier this session) — verified, public-visibility email on the user's GitHub account.

### Step 4 — Run `git filter-repo`

Single pass with two transformations:

```
git filter-repo \
  --mailmap .mailmap \
  --message-callback '
import re
patterns = [
    rb"^Co-authored-by:\s*Claude.*?<noreply@anthropic\.com>\s*$",
    rb"^Co-authored-by:\s*Cursor Agent\s*<cursoragent@cursor\.com>\s*$",
    rb"^Co-authored-by:\s*Copilot\s*<\d+\+Copilot@users\.noreply\.github\.com>\s*$",
]
lines = message.split(b"\n")
kept = []
for line in lines:
    if any(re.match(p, line, re.IGNORECASE) for p in patterns):
        continue
    kept.append(line)
return b"\n".join(kept)
'
```

This rewrites the 4 bot-author commits (mailmap path) AND strips the agent Co-Authored-By trailer lines from all 38 affected commits in one pass. No file content changes.

### Step 5 — Force-push main

```
git push --force-with-lease origin main
```

`--force-with-lease` aborts if origin moved unexpectedly (safer than plain `--force`).

### Step 6 — Reconcile other branches

Three local branches diverged from new main. Two options per branch:
- Rebase onto new main: `git rebase main <branch>` (preserves work, may have conflicts)
- Discard if abandoned: `git branch -D <branch>` (use backup if recovery needed)

## Outcome (intended)

- github.com contributors page on `main` collapses to: `jakobtfaber` (146 + 4 = 150) + `jakobtfaber-2` (1, kept by user choice) + possibly the `jakob.faber@nanograv.org` ghost (86 commits, separate issue not addressed by this plan).
- Cursor Agent and Copilot disappear from contributors graph for `main`.
- Claude Co-Authored-By trailers removed from 36 commit messages; Claude no longer credited via avatars on PR/commit pages.
- All commit content (file diffs) unchanged; only metadata + trailer lines change.
- Every commit on `main` gets a new SHA.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Force-push overwrites collaborator work | `--force-with-lease`; backup branches pushed to origin first; reflog locally |
| Open PRs (if any) become invalid | None known; user solo-maintains; check `gh pr list` first |
| Other local branches diverge | Explicit rebase-or-delete step, with backups pushed to origin |
| Mailmap regex matches more than intended | Test with `git log --use-mailmap` before filter-repo run; review filter-repo's preview |
| Co-Authored-By regex over/under-strips | Run filter-repo in a separate clone first as dry-run |
| 1M-context Claude variant trailer not matched | Pattern uses `Claude.*?<noreply@anthropic.com>` which catches all Claude variants; verify with grep before run |
| Verified-but-private gh007 push block reappears | Caltech email is verified+public-visibility on user account; should pass, but reproducible test before force-push |

## Questions to review

1. Is `git-filter-repo` with `--mailmap` + `--message-callback` the right tool for this combined operation, or is there a simpler path (e.g. interactive rebase with a script)?
2. Is the message-callback regex correct for stripping ONLY the agent Co-Authored-By lines and not affecting other commit message content?
3. Are there safety steps missing? (Specifically: dry-run mechanism, verification before force-push, recovery procedure if origin force-push corrupts things.)
4. Is `--force-with-lease` sufficient, or should we use a different push strategy?
5. Will `filter-repo --mailmap` apply to BOTH author and committer fields, or only author? If only author, do we also need to fix committer?
6. Does filter-repo handle the case where the same commit's message is edited by message-callback AND its author is rewritten by mailmap?
7. Is the proposed branch-reconciliation step (rebase or discard) the right approach, or is there a cleaner pattern?
8. Anything else missed?
