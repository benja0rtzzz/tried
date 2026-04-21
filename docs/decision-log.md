# Decision Log

Lightweight ADR (Architecture Decision Record) log. Write an entry **before** changing schema, tolerance policy, closed vocabularies, or any other load-bearing invariant. Wait for team sign-off, then update the implementation.

## Entry format

```markdown
## YYYY-MM-DD — Short title
**Status:** proposed | accepted | rejected
**Problem:** one sentence describing what needs to change and why.
**Decision:** what we are doing.
**Consequences:** what breaks, what improves, what we lose.
```

---

## 2026-04-21 — Initial project structure
**Status:** accepted
**Problem:** Project needed a defined repo layout, UV workspace split, and documentation structure before coding begins.
**Decision:** UV workspace with three packages (`shared`, `orchestrator`, `verification`). Docs split into per-topic files. Schema and tolerance policy designated as locked files requiring log entry + team sign-off before any change.
**Consequences:** Clear ownership boundaries per machine. Schema/tolerance changes are intentionally friction-heavy to protect experiment validity.
