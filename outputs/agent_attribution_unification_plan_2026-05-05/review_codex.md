## Technical correctness

- Lines 49-52: mailmap syntax is valid, but line 47 says “add ... alongside” while existing `.mailmap` already maps Cursor Agent to a different canonical email. Replace it or use a temporary external mailmap.
- Lines 61-76: callback is bytes-correct: `message` is bytes, regexes are bytes, and it returns bytes.
- Lines 65-69: regexes match current agent trailer variants found on `main`: Claude Opus/Sonnet variants, Cursor Agent, and Copilot. They intentionally leave Jakob co-author trailers.
- Line 62: `--mailmap` rewrites matching author, committer, and tagger identities, not only author. It will not rewrite unmatched `GitHub <noreply@github.com>` committers.
- Lines 56-80: unsafe as written in this working clone. `git-filter-repo` normally expects a fresh clone, rewrites refs/history, writes `.git/filter-repo/*`, and removes `origin`; line 85 may fail unless `origin` is re-added.
- Lines 80-81: “No file content changes” conflicts with Step 3 if `.mailmap` is committed.

## Safety

- Lines 34-43: backup branches are useful but insufficient. Make a local mirror/bundle backup of all refs too; do not rely on reflog.
- Line 85: prefer explicit lease: `git push --force-with-lease=main:<old-origin-main-sha> origin HEAD:main`.
- Test in a fresh `--no-local` clone first, then push a preview branch before touching `main`.
- Signed commits are a real issue: I found signed commits in `main`; rewriting invalidates those signatures.
- Line 17 misses local branches `2c195c6a`, `a50bbad9`, and remote-tracking branches. Decide all-refs vs main-only explicitly.

## Better alternatives

- `.mailmap` alone avoids rewrite but will not fix GitHub contributors page.
- No normal repo-local GitHub setting reliably filters selected contributors from the contributors graph.
- Split testing: mailmap-only rewrite first, trailer stripping second. Final production can still be one pass once verified.
- Consider `--refs main` if only default-branch contributors matter, but document that old refs remain.

## Edge cases

- Tags: local tags are empty, but remote tags could not be checked from this sandbox. Verify remote tags before rewriting.
- Old SHAs may exist in PRs, issue comments, CI logs, releases, notebooks, and docs; archive `commit-map`.
- Backup branches in the same GitHub repo preserve old commits and may preserve old attribution surfaces.
- Do not fold in the Nanograv-email ghost unless scope is explicitly expanded.

## Verdict (proceed / revise / abort)

Revise. Mechanism is plausible, but the operational plan is not safe enough until it uses a fresh clone, resolves the `.mailmap` ambiguity, accounts for filter-repo’s ref/origin cleanup, checks tags/branch protection/signatures, and force-pushes with an explicit lease after preview verification.

Sources checked: [`git-filter-repo` docs](https://raw.githubusercontent.com/newren/git-filter-repo/main/Documentation/git-filter-repo.txt), [`gitmailmap` docs](https://git-scm.com/docs/gitmailmap).
## Technical correctness

- Lines 49-52: mailmap syntax is valid, but line 47 says “add ... alongside” while existing `.mailmap` already maps Cursor Agent to a different canonical email. Replace it or use a temporary external mailmap.
- Lines 61-76: callback is bytes-correct: `message` is bytes, regexes are bytes, and it returns bytes.
- Lines 65-69: regexes match current agent trailer variants found on `main`: Claude Opus/Sonnet variants, Cursor Agent, and Copilot. They intentionally leave Jakob co-author trailers.
- Line 62: `--mailmap` rewrites matching author, committer, and tagger identities, not only author. It will not rewrite unmatched `GitHub <noreply@github.com>` committers.
- Lines 56-80: unsafe as written in this working clone. `git-filter-repo` normally expects a fresh clone, rewrites refs/history, writes `.git/filter-repo/*`, and removes `origin`; line 85 may fail unless `origin` is re-added.
- Lines 80-81: “No file content changes” conflicts with Step 3 if `.mailmap` is committed.

## Safety

- Lines 34-43: backup branches are useful but insufficient. Make a local mirror/bundle backup of all refs too; do not rely on reflog.
- Line 85: prefer explicit lease: `git push --force-with-lease=main:<old-origin-main-sha> origin HEAD:main`.
- Test in a fresh `--no-local` clone first, then push a preview branch before touching `main`.
- Signed commits are a real issue: I found signed commits in `main`; rewriting invalidates those signatures.
- Line 17 misses local branches `2c195c6a`, `a50bbad9`, and remote-tracking branches. Decide all-refs vs main-only explicitly.

## Better alternatives

- `.mailmap` alone avoids rewrite but will not fix GitHub contributors page.
- No normal repo-local GitHub setting reliably filters selected contributors from the contributors graph.
- Split testing: mailmap-only rewrite first, trailer stripping second. Final production can still be one pass once verified.
- Consider `--refs main` if only default-branch contributors matter, but document that old refs remain.

## Edge cases

- Tags: local tags are empty, but remote tags could not be checked from this sandbox. Verify remote tags before rewriting.
- Old SHAs may exist in PRs, issue comments, CI logs, releases, notebooks, and docs; archive `commit-map`.
- Backup branches in the same GitHub repo preserve old commits and may preserve old attribution surfaces.
- Do not fold in the Nanograv-email ghost unless scope is explicitly expanded.

## Verdict (proceed / revise / abort)

Revise. Mechanism is plausible, but the operational plan is not safe enough until it uses a fresh clone, resolves the `.mailmap` ambiguity, accounts for filter-repo’s ref/origin cleanup, checks tags/branch protection/signatures, and force-pushes with an explicit lease after preview verification.

Sources checked: [`git-filter-repo` docs](https://raw.githubusercontent.com/newren/git-filter-repo/main/Documentation/git-filter-repo.txt), [`gitmailmap` docs](https://git-scm.com/docs/gitmailmap).
