# 🔒 Phase Gate Enforcement Architecture

> **Classification:** Security-Critical Infrastructure
> **Status:** Stable
> **Pattern Lineage:** circuit_breaker → label-enforce → openspace/phase-gates

---

## Executive Summary

A **fail-closed enforcement system** that makes it **technically impossible** to merge
out-of-phase code into the OpenSpace repository without explicit, audited admin override.

**Key guarantee:** If the enforcement system crashes, is misconfigured, or encounters any
unexpected state — the merge is BLOCKED, not allowed.

---

## 1. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ENFORCEMENT TOPOLOGY                             │
│                                                                         │
│  Developer creates PR                                                   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────┐     ┌──────────────────────────────┐              │
│  │  PR Event Fires  │────▶│  phase-enforce.yml (Action)  │              │
│  └─────────────────┘     │                              │              │
│                          │  LAYER 1: Issue Linkage      │              │
│                          │  ├─ PR body contains          │              │
│                          │  │  "Closes #N" or "Fixes #N"│              │
│                          │  └─ At least one linked issue │              │
│                          │                              │              │
│                          │  LAYER 2: Milestone Check     │              │
│                          │  ├─ Linked issue has milestone│              │
│                          │  └─ Milestone = valid phase   │              │
│                          │                              │              │
│                          │  LAYER 3: Phase Gate Check    │              │
│                          │  ├─ Load PHASE_DEPS graph     │              │
│                          │  ├─ For each prerequisite:    │              │
│                          │  │  └─ Is milestone 100%?     │              │
│                          │  └─ ALL prereqs must be met   │              │
│                          │                              │              │
│                          │  LAYER 4: Escape Hatch Check  │              │
│                          │  ├─ Has `emergency:bypass`?   │              │
│                          │  ├─ Bypass reason in PR body? │              │
│                          │  └─ Log override to audit     │              │
│                          │                              │              │
│                          │  LAYER 5: Audit Emission      │              │
│                          │  └─ PR comment with verdict   │              │
│                          └──────────────┬───────────────┘              │
│                                         │                               │
│                                         ▼                               │
│                          ┌──────────────────────────────┐              │
│                          │  GitHub Required Status Check │              │
│                          │                              │              │
│                          │  ✅ status = "success"        │              │
│                          │     → Merge button ENABLED    │              │
│                          │                              │              │
│                          │  ❌ status = "failure"        │              │
│                          │     → Merge button DISABLED   │              │
│                          │                              │              │
│                          │  ⚫ status = (missing/pending)│              │
│                          │     → Merge button DISABLED   │  ← FAIL-CLOSED
│                          │     (Action crashed or hung)  │              │
│                          └──────────────────────────────┘              │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    BRANCH PROTECTION (Layer 0)                   │   │
│  │                                                                   │   │
│  │  main branch:                                                     │   │
│  │  ├─ Require PR before merging (no direct push)                   │   │
│  │  ├─ Require status check: "Phase Gate Enforcement" (STRICT)      │   │
│  │  ├─ Require conversation resolution                               │   │
│  │  ├─ Do NOT allow bypassing (even for admins — see escape hatch)  │   │
│  │  └─ Restrict pushes: only github-actions[bot]                     │   │
│  │                                                                   │   │
│  │  WHY: Even admins can't push to main. The ONLY path to main is  │   │
│  │  through a PR that passes the enforcement check.                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Phase Dependency Graph

```
                    P0 (Foundation)
                   / \
                  /   \
                 ▼     ▼
    P1 (Core A)        P2 (Core B)
       │                  │
       ▼                  │
    P3 (Extensions)       │
       │                  │
       ▼                  │
    P4 (Integration)      │
       │                  │
       ▼                  ▼
       P5 (Convergence) ◀─┘
        │
        ▼
       P6 (Hardening)
        │
        ▼
       P7 (Release)
```

**Encoded as configuration:**

```yaml
# .github/phase-config.yml
phases:
  "Phase 0 — Emergency Hardening":
    prerequisites: []            # No prerequisites — always open
  "Phase 1 — Foundation Architecture":
    prerequisites: ["Phase 0 — Emergency Hardening"]
  "Phase 2 — Smart Sandbox":
    prerequisites: ["Phase 0 — Emergency Hardening"]
  "Phase 3 — Extract store.py":
    prerequisites: ["Phase 1 — Foundation Architecture"]
  "Phase 4 — Extract tool_layer + mcp_server":
    prerequisites: ["Phase 3 — Extract store.py"]
  "Phase 5 — Extract evolver + grounding":
    prerequisites:
      - "Phase 2 — Smart Sandbox"
      - "Phase 4 — Extract tool_layer + mcp_server"
  "Phase 6 — Production Readiness":
    prerequisites: ["Phase 5 — Extract evolver + grounding"]
  "Phase 7 — Enforcement & Launch":
    prerequisites: ["Phase 6 — Production Readiness"]

# Milestone names MUST match these phase names exactly.
# This is the single source of truth for phase ordering.
```

## 3. Enforcement Decision Matrix

| Condition | Result | Status Check | Audit |
|-----------|--------|-------------|-------|
| PR links issue(s) in completed-prereq phase | ✅ PASS | `success` | Green comment |
| PR links issue(s) in Phase 0 (no prereqs) | ✅ PASS | `success` | Green comment |
| PR has `emergency:bypass` label + reason | ⚠️ BYPASS | `success` | Orange comment with reason |
| PR links no issues | ❌ FAIL | `failure` | Red comment: "Link an issue" |
| PR links issue with no milestone | ❌ FAIL | `failure` | Red comment: "Add to milestone" |
| PR links issue in milestone with unmet prereqs | ❌ FAIL | `failure` | Red comment: lists blockers |
| PR has `emergency:bypass` but no reason | ❌ FAIL | `failure` | Red comment: "Provide bypass reason" |
| Enforcement Action crashes | ⬛ MISSING | BLOCKED | No comment (investigate) |
| GitHub API rate limited | ❌ FAIL | `failure` | Red comment: "Retry in N min" |

**The last two rows are the fail-closed guarantee.** Missing status = blocked.

## 4. Layer Details

### Layer 0: Branch Protection (GitHub Settings)

Configure via Settings → Branches → Branch protection rules → `main`:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Require a pull request before merging | ✅ | No direct pushes |
| Require approvals | 0 (solo dev) | Solo dev doesn't need self-approval |
| Dismiss stale pull request approvals | ✅ | If you add reviewers later |
| Require status checks to pass | ✅ **CRITICAL** | This is the enforcement point |
| Status check: `Phase Gate Enforcement` | ✅ Required | Must match workflow job name |
| Require branches to be up to date | ✅ | Prevents stale merges |
| Require conversation resolution | ✅ | Forces addressing review comments |
| Do not allow bypassing the above settings | ✅ | Even admins must use escape hatch |
| Restrict who can push to matching branches | github-actions[bot] | Belt and suspenders |

> **CRITICAL:** "Do not allow bypassing" is what makes this fail-closed for admins too.
> The escape hatch is through the `emergency:bypass` label, not through admin override
> of branch protection. This way every bypass is audited.

### Layer 1: PR → Issue Linkage

```
PR body or title must contain:
  - "Closes #N" or "Fixes #N" or "Resolves #N"
  - At least ONE valid issue reference

Why not just "mentions #N"?
  - "Closes" creates a formal link GitHub tracks
  - Prevents gaming by mentioning random issues
  - Auto-closes issue on merge (existing auto-close.yml behavior)
```

### Layer 2: Issue → Milestone Mapping

```
Every linked issue must belong to a milestone.
The milestone name must match a key in phase-config.yml.

Why milestone, not labels?
  - Milestones have completion tracking (open/closed counts)
  - Milestones are a single assignment (can't be in two phases)
  - GitHub API provides milestone.open_issues natively
  - Labels are used for other things (type:, priority:, etc.)
```

### Layer 3: Phase Gate Validation

```python
# Pseudocode for the core logic
def check_phase_gate(issue_milestone, phase_config, all_milestones):
    phase = phase_config[issue_milestone]
    
    for prereq_name in phase.prerequisites:
        prereq_milestone = find_milestone(prereq_name, all_milestones)
        
        if prereq_milestone is None:
            FAIL("Prerequisite milestone '{prereq_name}' not found")
        
        if prereq_milestone.open_issues > 0:
            FAIL(f"Phase '{prereq_name}' has {prereq_milestone.open_issues} open issues")
    
    PASS("All prerequisite phases complete")
```

### Layer 4: Escape Hatch

The escape hatch uses a **two-key mechanism** (inspired by nuclear launch protocols):

1. **Key 1:** `emergency:bypass` label on the PR (admin-only label)
2. **Key 2:** PR body contains `## Bypass Reason\n<reason text>`

Both are required. If only the label exists without a reason, the check still fails.

```
Emergency bypass is for:
  ✅ Critical security patches
  ✅ Production-down hotfixes
  ✅ Phase gate enforcement bugs
  ✅ External dependency deadlines

Emergency bypass is NOT for:
  ❌ "I'm almost done with this phase"
  ❌ "This is a small change"
  ❌ "I'll fix the issue later"
```

### Layer 5: Audit Trail

Every enforcement decision produces a **structured PR comment**:

```markdown
## 🔒 Phase Gate Enforcement — [PASS|FAIL|BYPASS]

| Check | Result | Detail |
|-------|--------|--------|
| Issue linkage | ✅/❌ | #42, #43 |
| Milestone assigned | ✅/❌ | Phase 2 — Core B |
| Phase prerequisites | ✅/❌ | Phase 0 ✅, Phase 1 ⏳ (3 open) |
| Emergency bypass | ➖/⚠️ | N/A or "reason text" |

**Verdict:** MERGE ALLOWED / MERGE BLOCKED
**Timestamp:** 2025-01-15T10:30:00Z
**Run:** [Actions link]
```

## 5. What CAN'T Be Enforced This Way?

### Hard Limits of GitHub Actions Enforcement

| Gap | Description | Mitigation |
|-----|-------------|------------|
| **Issue quality** | Can't verify the issue actually describes real work | Human review / triage process |
| **Milestone accuracy** | Can't verify an issue is in the RIGHT milestone | Code review + milestone audits |
| **Code correctness** | Phase gate ≠ code quality | Separate CI/CD pipeline (tests, lint) |
| **Partial completion** | Issue marked "closed" but work is incomplete | PR review discipline |
| **Cross-repo deps** | If phases span multiple repos | Extend to org-level workflows |
| **Retroactive edits** | Someone could close prerequisite issues without doing work | Progress dashboard catches anomalies |
| **GitHub outage** | If GitHub Actions is down, no status = no merge (GOOD — fail-closed) | Wait for GitHub to recover |
| **Rate limits** | 1000 API calls/hour for Actions | Batch API calls; retry with backoff |
| **Race conditions** | Two PRs pass gate simultaneously, one re-opens prereq issues | Re-run check on push to PR branch |

### Branch Strategy Tradeoff Analysis

| Strategy | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Single main + status checks** | Simple, clear, one source of truth | All enforcement is in the Action | ✅ **Recommended** |
| **Branch per phase** | Physical isolation, easy to reason about | Complex merge choreography, 8 branches, merge conflicts | ❌ Overkill for solo/small team |
| **Hybrid (main + release branches)** | Release isolation | Unnecessary until Phase 7 | ❌ Premature complexity |

**Recommendation: Single `main` branch with enforcement via required status checks.**

For a solo dev or small team, branch-per-phase creates merge conflict hell across 8 long-lived
branches. The enforcement Action provides equivalent protection with zero branch management overhead.

## 6. Failure Mode Analysis

Inspired by the circuit breaker pattern: every failure mode is enumerated and handled.

| Failure | Category | System Behavior | Recovery |
|---------|----------|-----------------|----------|
| Action YAML syntax error | Crash | No status reported → BLOCKED | Fix YAML, re-push |
| `actions/github-script` OOM | Crash | No status reported → BLOCKED | Optimize script |
| GitHub API 500 | Transient | Retry 3x with backoff → FAIL if persists | Wait, re-run |
| GitHub API 403 (rate limit) | Transient | FAIL with "rate limited" message | Wait for reset |
| phase-config.yml missing | Config | FAIL with clear error message | Add config file |
| phase-config.yml malformed | Config | FAIL with validation error | Fix YAML |
| Milestone name mismatch | Config | FAIL: "milestone X not in config" | Align names |
| PR references deleted issue | Edge case | FAIL: "issue #N not found" | Fix PR body |
| Circular dependency in config | Config | FAIL: "circular dep detected" | Fix config |
| New phase added, config not updated | Config | FAIL: "milestone X not in config" | Update config |

**Key insight from circuit breaker design:** The circuit breaker pattern (HEALTHY → RETRY → CIRCUIT_OPEN)
maps to our GitHub Actions retry strategy. We retry API calls 3x with backoff before failing.
Unlike a full circuit breaker, we don't need AWAITING_USER because the "user" (developer) is already
present — they see the failed check and can re-run it.

## 7. Security Considerations

| Threat | Vector | Mitigation |
|--------|--------|------------|
| Modify enforcement workflow | Push to `.github/workflows/` | Branch protection prevents direct push; PR still needs to pass OLD enforcement |
| Create bypass label without authorization | Label the PR | Make `emergency:bypass` a **protected label** (org-level, or enforce via label enforcement workflow) |
| Close prerequisite issues to fake completion | Close issues without work | Progress dashboard shows velocity anomalies; issues without PRs are suspicious |
| Fork + PR from fork | Different branch protection | Fork PRs still trigger the same required status check |
| Workflow_dispatch bypass | Trigger Action manually | Enforcement is on `pull_request` trigger, not `workflow_dispatch` |
| Delete and recreate main branch | Nuclear option | Repo admin only; audit log catches this |

## 8. Implementation Checklist

- [ ] Create `.github/phase-config.yml` with phase dependency graph
- [ ] Create `.github/workflows/phase-enforce.yml` (the enforcement Action)
- [ ] Configure branch protection on `main` per Layer 0 settings
- [ ] Create `emergency:bypass` label (restricted to admins)
- [ ] Create `phase-enforce-audit` label for tracking audit comments
- [ ] Test with a PR that SHOULD be blocked (Phase 1 issue, Phase 0 incomplete)
- [ ] Test with a PR that SHOULD pass (Phase 0 issue, no prereqs)
- [ ] Test the escape hatch (add label + reason → should pass)
- [ ] Test crash behavior (syntax error in phase-config.yml → should block)
- [ ] Document for contributors in README

---

*This document is the single source of truth for the enforcement architecture.
Changes to the enforcement system MUST update this document first.*
