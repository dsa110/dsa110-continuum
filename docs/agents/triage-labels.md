# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the label strings used on `dsa110/dsa110-continuum`.

| Canonical role     | Label on this repo  | Meaning                                                     |
| ------------------ | ------------------- | ----------------------------------------------------------- |
| `needs-triage`     | `needs-triage`      | Maintainer needs to evaluate this issue                     |
| `needs-info`       | `needs-info`        | Waiting on reporter for more information                    |
| `ready-for-agent`  | `ready-for-agent`   | Fully specified, ready for an AFK agent to pick up cold     |
| `ready-for-human`  | `ready-for-human`   | Requires human implementation (calibration intuition, etc.) |
| `wontfix`          | `wontfix`           | Will not be actioned                                        |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding label string from this table.

These are *workflow* labels — orthogonal to *type* labels like `bug`, `enhancement`, `documentation`. The triage skill only manages this column.
