# Planner Phase — Implementation

**Purpose:** These Taskplane packets implement approved planner contract slices.

## Rules

- The planner contract is authoritative: use the change proposal, design, and relevant delta spec as the source of truth.
- Implement only the approved requested delta for the capability assigned to the packet.
- Preserve the approved interface constraints, preservation constraints, and proof obligations.
- If execution reveals a contract defect, capture the discovery and escalate for planner reopening instead of broadening scope silently.
- Do not write the final whole-change conformance verdict from an implementation packet.

## Expected Outcome

- Code, tests, and docs for the assigned capability slice are complete.
- Repo gates required by the approved contract pass.
- The packet stays within the approved edit surface unless an amendment is explicitly required.
