# OpenSpace Upgrade — Agent Operating Instructions
# ==================================================
# Auto-loaded by Copilot CLI, Claude Code, and Codex on session start.
# Protected by CODEOWNERS — changes require admin review.
#
# This file is the PRIME DIRECTIVE for autonomous epic execution.
# It is NOT a suggestion. It is the loop.

# ─── IDENTITY ───────────────────────────────────────────────────────
# You are working on the OpenSpace upgrade project.
# Your job: execute the epic cycle until all phases (P3–P7) are complete.

# ─── ON SESSION START (MANDATORY) ──────────────────────────────────

## 1. Read State
- Parse `RUNBOOK.yaml` in repo root → find the next epic with `status: pending`
- Check `lock.active_claim` → if another session claimed an epic and TTL hasn't
  expired, do NOT claim any epic. Ask the user what to do (wait, override, or
  work on something else). Never silently pick a different epic.
- Read the latest handoff at `~/.copilot.working/sessions/` for this project
- Run `git status` and `python -m pytest --tb=short -q` to verify clean state
  (currently 1356+ tests passing)

## 2. Claim Epic
- Update `RUNBOOK.yaml`:
  - Set `lock.active_claim` to the epic ID
  - Set `lock.claimed_by` to `{date}-{session-id}`
  - Set `lock.claimed_at` to current ISO 8601 timestamp
  - Set the epic's `status` to `claimed`
- Do NOT commit this change — it's local working state only

## 3. Announce
```
Resuming OpenSpace upgrade.
Epic {id}: {title} (Phase {phase})
Last session: {handoff summary or "fresh start"}
Starting epic cycle.
```

# ─── EPIC CYCLE (MANDATORY FOR EVERY EPIC) ─────────────────────────

## Step 1: PLAN
- Use `brainstorming` skill if design decisions needed
- Use `writing-plans` skill to create implementation plan
- Create feature branch: `epic/{phase}.{number}-{name}`

## Step 2: BUILD
- Use `subagent-driven-development` skill with TDD
- RED → GREEN → REFACTOR for every change
- Commit frequently with descriptive messages

## Step 3: TEST
- Run full pytest suite: `python -m pytest --tb=short -q`
- ALL tests must pass (currently 1028+)
- If tests fail: fix before proceeding. Do NOT skip.

## Step 4: PUSH
- `git push origin {branch}`
- Create PR if not exists (link to GitHub issue, set milestone)

## Step 5: REVIEW — /8eyes
- Launch /8eyes multi-agent review (minimum 3 models)
- Collect findings from all roles

## Step 6: REVIEW — /collab
- Launch /collab multi-model buyoff (top-tier models per provider)
- Collect findings

## Step 7: FIX
- Address ALL findings from Step 5 + 6
- Re-run full test suite after fixes
- Commit fixes, push

## Step 8: RE-REVIEW (loop Steps 5–7)
- Track review round in `RUNBOOK.yaml` epic's `review_round` field
- **HARD CAP: 3 rounds maximum**
- If round 3 still has findings:
  1. `git stash` or `git reset` to pre-fix state
  2. Update epic status to `blocked` in RUNBOOK.yaml
  3. `/handoff` with detailed findings
  4. STOP — human intervention required
- If all roles PASS: proceed to Step 9

## Step 9: REQUEST MERGE
- Update epic status to `in_review` in RUNBOOK.yaml
- Ensure PR has: linked issue, milestone, passing CI
- **DO NOT merge yourself** — human approves the PR
- Wait for human approval, or `/handoff` and stop

## Step 10: POST-MERGE
- After human merges: the PR itself should include the RUNBOOK.yaml update
  setting the epic's `status` to `merged` (include in the epic branch, not main)
- Clear local lock: set `lock.active_claim` to null locally
- Do NOT commit RUNBOOK.yaml directly to main — all changes go through PR review
  (CODEOWNERS requires admin approval for RUNBOOK.yaml)

## Step 11: CONTINUE or STOP
- Check context health (`/health` or estimate)
- If context < 60% used: claim next epic, go to Step 1
- If context ≥ 60% used: `/handoff` and STOP (human restarts fresh session)
- If no more epics in current phase: check if next phase prerequisites are met
- If all P7 epics merged: **PROJECT COMPLETE** 🎉

# ─── CONTEXT MANAGEMENT ────────────────────────────────────────────

## When to /compact
- After completing a review round (before starting next round)
- After merging an epic (before starting next epic)

## When to /handoff and STOP
- Context ≥ 60% used
- Review round 3 still has findings (blocked)
- Design decision needed that requires human judgment
- Any error you can't resolve in 3 attempts
- Session has been running for 3+ hours

# ─── NON-NEGOTIABLE RULES ──────────────────────────────────────────

1. **No epic ships without /8eyes + /collab review** — both required, every time (honor system — CI does not enforce this; discipline is the gate)
2. **No merge without full test suite passing** — zero exceptions
3. **No skipping phases** — CI enforces via phase-config.yml
4. **No self-merge** — human clicks Approve (merge_policy: human_approval)
5. **3-round review cap** — abort + handoff, don't thrash
6. **Claim before building** — prevents concurrent session conflicts
7. **/handoff before context exhaustion** — never lose state
8. **Include RUNBOOK.yaml status update in epic PRs** — reviewed via CODEOWNERS

# ─── KNOWN LIMITATIONS ─────────────────────────────────────────────
# 1. /8eyes + /collab are honor-system — CI does not verify they ran
# 2. Crash before /handoff loses review state — git branch is the fallback
# 3. Lock mechanism is cooperative (YAML), not transactional
# 4. Handoff files lack integrity verification (no HMAC/signature)

# ─── FAILURE RECOVERY ──────────────────────────────────────────────

| Situation | Action |
|-----------|--------|
| Session crash before /handoff | Next session: read RUNBOOK.yaml + git state. Epic with `claimed` status and expired TTL → reclaim and continue. Review state may be lost — re-run /8eyes from the current branch state. |
| Tests fail after fix | Debug with `systematic-debugging` skill. 3 attempts max, then `/handoff`. |
| /8eyes or /collab unavailable | `/handoff` with note. Do NOT skip review. |
| Merge conflict with main | Rebase, re-test, re-push. If complex: `/handoff`. |
| Design decision needed | Ask user with `ask_user` tool. If user unavailable: `/handoff`. |
| CI blocks merge | Read CI error. Fix if phase/milestone issue. If unclear: `/handoff`. |

# ─── CREDENTIAL REQUIREMENTS ───────────────────────────────────────
# The agent credential MUST be:
# - A dedicated GitHub App or machine user (NOT a human's PAT)
# - Scoped to: contents:write on epic/* branches, pull-requests:write
# - NO admin privileges, NO bypass branch protection
# - NOT in escape_hatch.allowed_actors list
# - Short-lived tokens preferred (1hr max)
