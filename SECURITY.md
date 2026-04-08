# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 2.0.x   | ✅ Active support  |
| < 2.0   | ❌ Not supported   |

## Reporting a Vulnerability

If you discover a security vulnerability in Scion, please report it responsibly:

1. **Do NOT open a public GitHub issue.**
2. Email the maintainers at the address listed in the repository's GitHub Security Advisories tab.
3. Alternatively, use [GitHub's private vulnerability reporting](https://github.com/Deepfreezechill/scion/security/advisories/new).

### What to include

- Description of the vulnerability
- Steps to reproduce
- Impact assessment (RCE, data leak, DoS, etc.)
- Suggested fix (if any)

### Response timeline

- **Acknowledgment:** within 48 hours
- **Initial assessment:** within 7 days
- **Fix or mitigation:** within 30 days for critical/high severity

## Security Architecture

Scion employs defense-in-depth:

- **Bearer token + HMAC auth** on all MCP endpoints
- **Capability lease model** with tier-gated access, expiration, and revocation
- **5-layer ReviewGate** for skill evolution (path, extension, size, AST, lineage)
- **40+ AST blocklist patterns** preventing RCE, sandbox escape, and data exfiltration
- **Filesystem jailing** with TOCTOU race protection
- **Network SSRF protection** blocking loopback, link-local, and domain rebinding
- **E2B sandbox enforcement** as default execution environment
- **PII redaction** in structured logging
- **Dependency auditing** via `pip-audit` in CI

## Known Mitigations

- `litellm` pinned to `>=1.83.0,<2` — excludes PYSEC-2026-2 (supply-chain compromise in 1.82.7/1.82.8) and fixes CVE-2026-35029 + CVE-2026-35030
- All dependencies have upper-bound pins enforced by CI tests
- Cloud auto-import disabled by default (must be explicitly enabled)

## Deployment Security Boundaries

### local_server (VM-only component)

The `scion/local_server/` module is a Flask-based desktop automation server designed **exclusively for local VM environments** (e.g., OSWorld). It exposes intentionally permissive endpoints (`/run_python`, `/file`, `/upload_file`) for VM control.

**⚠️ Do NOT expose local_server to the internet or untrusted networks.** It must:
- Bind to `127.0.0.1` (loopback) only
- Run inside a VM or behind a firewall
- Never be deployed as a public-facing service

The MCP server (`scion-mcp`) is the production-safe entry point for agent integrations. It includes bearer token authentication, rate limiting, and SSRF protection.
