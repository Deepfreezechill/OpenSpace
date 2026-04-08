#!/usr/bin/env bash
# ============================================================================
# configure-branch-protection.sh
# ============================================================================
#
# Configures branch protection rules for the OpenSpace repo.
# Run this ONCE after pushing the enforcement workflow.
#
# Prerequisites:
#   - GitHub CLI (gh) installed and authenticated
#   - Admin access to the repository
#   - phase-enforce.yml must have run at least once (for status check to exist)
#
# Usage:
#   chmod +x configure-branch-protection.sh
#   ./configure-branch-protection.sh <owner> <repo>
#
# Example:
#   ./configure-branch-protection.sh Deepfreezechill OpenSpace
# ============================================================================

set -euo pipefail

OWNER="${1:?Usage: $0 <owner> <repo>}"
REPO="${2:?Usage: $0 <owner> <repo>}"

echo "🔒 Configuring branch protection for ${OWNER}/${REPO}:main"
echo ""

# Step 1: Create the emergency:bypass label if it doesn't exist
echo "📋 Creating emergency:bypass label..."
gh label create "emergency:bypass" \
  --repo "${OWNER}/${REPO}" \
  --description "Emergency phase gate bypass — requires audit reason" \
  --color "FF0000" \
  2>/dev/null || echo "  (label already exists)"

# Step 2: Create audit label
echo "📋 Creating audit label..."
gh label create "audit" \
  --repo "${OWNER}/${REPO}" \
  --description "Enforcement audit trail record" \
  --color "FFA500" \
  2>/dev/null || echo "  (label already exists)"

# Step 3: Configure branch protection
# Note: gh api is used because `gh branch-protection` doesn't support all options
echo ""
echo "🛡️ Configuring branch protection on 'main'..."
echo ""
echo "   Required settings (configure manually in GitHub Settings → Branches):"
echo ""
echo "   ┌─────────────────────────────────────────────────────────────────┐"
echo "   │  Branch protection rule: main                                   │"
echo "   │                                                                 │"
echo "   │  [✅] Require a pull request before merging                     │"
echo "   │       Required approving reviews: 0 (solo dev)                  │"
echo "   │       [✅] Dismiss stale pull request approvals when new        │"
echo "   │            commits are pushed                                   │"
echo "   │                                                                 │"
echo "   │  [✅] Require status checks to pass before merging              │"
echo "   │       [✅] Require branches to be up to date before merging     │"
echo "   │       Status checks that are required:                          │"
echo "   │         → Phase Gate Enforcement                                │"
echo "   │                                                                 │"
echo "   │  [✅] Require conversation resolution before merging            │"
echo "   │                                                                 │"
echo "   │  [✅] Do not allow bypassing the above settings                 │"
echo "   │       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^              │"
echo "   │       THIS IS THE CRITICAL SETTING FOR FAIL-CLOSED             │"
echo "   │                                                                 │"
echo "   │  [✅] Restrict who can push to matching branches                │"
echo "   │       → github-actions[bot]  (for automated merges only)       │"
echo "   │                                                                 │"
echo "   └─────────────────────────────────────────────────────────────────┘"
echo ""

# Step 4: API-based configuration (what we can automate)
echo "🔧 Applying API-based branch protection..."

# Note: The GitHub REST API for branch protection requires the status check name
# to have been reported at least once. Push the workflow and run it on a test PR first.
gh api -X PUT "repos/${OWNER}/${REPO}/branches/main/protection" \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["Phase Gate Enforcement"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 0
  },
  "restrictions": null,
  "required_conversation_resolution": true
}
EOF

echo ""
echo "✅ Branch protection configured!"
echo ""
echo "⚠️  MANUAL STEP REQUIRED:"
echo "   Go to Settings → Branches → main → Edit"
echo "   Check: 'Do not allow bypassing the above settings'"
echo "   (This setting cannot be configured via API)"
echo ""
echo "🧪 TEST THE SETUP:"
echo "   1. Create a test PR that links to a Phase 1 issue (Phase 0 incomplete)"
echo "   2. Verify the merge button is DISABLED"
echo "   3. Add 'emergency:bypass' label + '## Bypass Reason' section"
echo "   4. Verify the merge button is ENABLED"
echo "   5. Remove the label, verify DISABLED again"
echo "   6. Close the test PR"
