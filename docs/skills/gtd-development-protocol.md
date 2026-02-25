---
name: gtd-development-protocol
description: Enforces "Getting Things Done" (GTD) protocols for agentic development. The agent must process inputs into Projects and Next Actions, maintaining a clear "Definition of Done" and "Next Action" at all times.
---

# GTD Development Protocol

## Core Philosophy
The agent must maintain a "Mind Like Water" state. To achieve this, nothing is kept in "memory" or strictly as a vague intent. Everything is **Captured**, **Processed**, **Organized**, **Reviewed**, and **Executed**.

## The Processing Algorithm
Before starting ANY work (changing files, running commands), the agent MUST clarify:
1.  **What is the Desired Outcome?** (The precise state that equals "Done")
2.  **What is the Next Action?** (The immediate next physical/digital step)

## Rules of Engagement

### 1. Desired Outcome Definition
You cannot proceed with a task unless you have defined the **Desired Outcome**.
- **Bad**: "Fix the bug."
- **Good**: "The `test_calibration_solver` passes without raising `LinAlgError`."

### 2. Next-Action Thinking
All items in your `task.md` or internal plan must be **Next Actions**, not Projects.
- A **Project** is any outcome requiring >1 step.
- A **Next Action** is a single, atomic, physical/digital action.
    - *Wrong*: "Implement logging." (This is a Project)
    - *Right*: "Create `logging_config.py` in `backend/src/core/`." (This is a Next Action)

### 3. Contexts
When organizing tasks, apply these contexts:
- **@Code**: Requires editing text files.
- **@Terminal**: Requires running CLI commands.
- **@Review**: Requires USER feedback or approval.
- **@Waiting**: Blocked by a long-running process or external dependency.

### 4. The 2-Minute Rule (Turbo Mode)
If a Next Action takes < 2 minutes (or very few tokens) to execute:
- **DO IT NOW**.
- Do not add it to a list. Do not wait for extensive planning.
- Use `// turbo` in workflows to auto-execute these.

### 5. Open Loops
If you encounter a `TODO`, `FIXME`, or a bug that is NOT your current focus:
- **Do NOT** fix it immediately (unless it blocks you).
- **CAPTURE** it. Add it to a "Parking Lot" or "Inbox" section in `task.md`.
- Stay focused on the current Next Action.

## Interaction with Task Boundary
When calling `task_boundary`:
- **TaskName**: Should reflect the **Project**.
- **TaskStatus**: Should reflect the **Next Action**.

---
## Cheat Sheet: Processing Input
When the User gives you a vague request (e.g., "Improve the docs"):
1.  Ask: "What is the specific component?" (Clarify)
2.  Ask: "Who is the audience?" (Context)
3.  Define: "Outcome = Documentation for `gpu_gridding.py` updated to include new parameters."
4.  Action: "Read `gpu_gridding.py` to identify missing params."
