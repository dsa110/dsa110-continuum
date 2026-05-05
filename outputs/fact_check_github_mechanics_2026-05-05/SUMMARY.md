# Fact-check summary — GitHub mechanics claims

Date: 2026-05-05
Source: Perplexity-MCP fact-checker, 9 claims made earlier in the conversation about GitHub commit-attribution and push-protection mechanics.

| # | Claim (paraphrased) | Verdict | Correction |
|---|---|---|---|
| 1 | `.mailmap` is purely local; github.com IGNORES it. | MOSTLY-FALSE | Practical effect right, framing too absolute. github.com doesn't *process* `.mailmap`, but the right framing is: github.com attributes commits by matching raw author email against registered emails on each account; `.mailmap` is independent. |
| 2 | Multiple verified emails on one account collapse to a single contributor entry. | MISLEADING | True per-commit attribution rule, but the contributors graph has caveats: top-100 only, default-branch only, excludes merge/empty commits. Same-user dedup is the *typical* outcome but not guaranteed in every case. |
| 3 | No-reply alias `<id>+username@users.noreply.github.com` is bound to one GitHub account. | MOSTLY-TRUE | Documented as such; GitHub says it cannot be unlinked or reattributed. Phrasing "claimed by another account" is slightly broader than the docs. |
| 4 | Co-Authored-By trailers do NOT add the trailer-named user to the contributors graph. | **MISLEADING** | **Wrong as I stated it.** Per GitHub docs, co-authored commits *can* count toward profile contributions and the contributors graph if the trailer email is verified on the co-author's account. |
| 5 | Only one email at a time can have `visibility=public`; field marks the public profile email, not "anyone can see this." | TRUE | Correct. |
| 6 | "Block command line pushes that expose my email" setting is what triggers GH007. | TRUE | Correct. Distinct from per-email visibility. |
| 7 | `git rebase --exec 'git commit --amend --reset-author --no-edit'` resets author name/email + author date to current config + now. | TRUE on substance | Each sub-claim verified by Git docs; the fact-checker's harsh "FALSE overall" rating is about phrasing (it conflates what `rebase` vs `commit --amend` do), not factual error. |
| 8 | Caltech-style email with `visibility=public` is exempt from GH007. | **MOSTLY-FALSE** | **Wrong as I stated it.** GitHub's block is based on whether the email is *marked private on your account*, not on its visibility-public flag. A verified, public-visibility email can still trigger GH007 depending on account configuration. |
| 9 | Co-Authored-By trailers show as avatars but don't add to contributors graph. | **FALSE** | **Wrong as I stated it.** Co-authored commits can appear in contribution graphs and repo statistics, not just as avatar attribution. Same correction as #4. |

## Material corrections to my previous advice

1. **Co-Authored-By trailers DO count toward github.com contribution graphs**, provided the trailer email is verified on the co-author's GitHub account. So the dozens of `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` trailers across this repo's history *could* be visually-counted contributions for whatever GitHub account (if any) has that anthropic noreply registered — likely none, so they're "ghost" contributions. But for *future* tooling that uses your no-reply alias as the Co-Authored-By email, those would credit to your account.

2. **Setting an email's per-email visibility to "public" does NOT guarantee push goes through.** The GH007 block is gated on whether the email is treated as *private on the account*, which is a separate concept. The Caltech address may still be subject to the block depending on account-level privacy settings. The push that just succeeded with `jfaber@caltech.edu` worked because *all the right toggles aligned*, not specifically because we set its visibility=public.

3. **`.mailmap` framing**: instead of saying "github.com ignores `.mailmap`," the accurate statement is "github.com attributes commits by raw author email matched against registered emails on each account; `.mailmap` is a Git-side display mapping that operates independently." Same practical outcome, but more defensible.

## Implications for the contributors-list unification goal

- The Cursor Agent + Copilot raw-author commits are still the targets to fix on github.com (rewrite or future-config).
- The Co-Authored-By trailers in commit messages are *also* potentially relevant — if you've authored commits with `Co-Authored-By: Claude` etc., those trailers may be creating ghost contributors. If they don't appear in the github.com contributors page, it's because no account has the trailer's email verified; if any *do* appear, they'd need addressing too.

## Sources by claim

JSON-formatted per-claim outputs alongside this file:
- `01_mailmap_local_only.json` … `09_coauthored_avatars_not_graph.json`

Each contains the full Perplexity response and source URLs.
