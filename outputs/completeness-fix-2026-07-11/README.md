# Coverage-aware completeness fix — Claude Fable 5 dispatch

## Auth gotcha

If `ANTHROPIC_API_KEY` is set in the environment, Claude Code prefers that
pay-as-you-go key and can report `Credit balance is too low` even when a Max
subscription OAuth login has Fable usage left.

Use OAuth:

```bash
env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
  claude auth status
# expect: authMethod oauth_token, and NO apiKeySource: ANTHROPIC_API_KEY
```

## Re-dispatch

```bash
./outputs/completeness-fix-2026-07-11/dispatch-claude-fable5.sh
```

Workflow: `ai-research-workflows` researching → planning → implementing → validating.
