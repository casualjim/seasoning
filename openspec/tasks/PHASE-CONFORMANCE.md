# Planner Phase — Conformance

**Purpose:** These Taskplane packets verify an assembled change against the approved planner contract.

## Rules

- Read the full approved contract before deciding the verdict.
- Evaluate the change against proposal intent, design constraints, delta specs, and proof obligations.
- Write findings and the explicit verdict to the change conformance report.
- Do not implement fixes directly from the conformance packet.
- If a fix is possible inside the approved contract, route it to remediation work. If the contract itself must change, route it to planner reopening.

## Disposition Model

- `LOG_ONLY`
- `INLINE_REVISE`
- `REMEDIATION_TASK`
- `REOPEN_PLANNING`
- `ESCALATE_HUMAN`
- `ARCHIVE_READY`

## Expected Outcome

- The change has a canonical conformance report with evidence and an explicit verdict.
- Archive happens only after an `ARCHIVE_READY` verdict.
