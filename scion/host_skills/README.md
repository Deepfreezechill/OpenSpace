# Host Skills Integration Guide

This guide covers **agent-specific setup** for integrating Scion. For installation and general concepts, see the [main README](../../README.md#-quick-start).

**Pick your agent:**

| Agent | Setup Guide |
|------------|-------------|
| **[nanobot](https://github.com/HKUDS/nanobot)** | [Setup for nanobot](#setup-for-nanobot) |
| **[openclaw](https://github.com/openclaw/openclaw)** | [Setup for openclaw](#setup-for-openclaw) |
| **Other agents** | Follow the [generic setup](../../README.md#-path-a-for-your-agent) in the main README |

---

## Setup for nanobot

### 1. Copy host skills

```bash
cp -r host_skills/skill-discovery/ /path/to/nanobot/nanobot/skills/
cp -r host_skills/delegate-task/ /path/to/nanobot/nanobot/skills/
```

### 2. Add MCP server to `~/.nanobot/config.json`

```json
{
  "tools": {
    "mcpServers": {
      "scion": {
        "command": "scion-mcp",
        "toolTimeout": 1200,
        "env": {
          "SCION_HOST_SKILL_DIRS": "/path/to/nanobot/nanobot/skills",
          "SCION_WORKSPACE": "/path/to/Scion",
          "SCION_API_KEY": "sk-xxx"
        }
      }
    }
  }
}
```

> [!TIP]
> LLM credentials are auto-detected from nanobot's `providers.*` config — no need to set `SCION_LLM_API_KEY`.

---

## Setup for openclaw

### 1. Copy host skills

```bash
cp -r host_skills/skill-discovery/ /path/to/openclaw/skills/
cp -r host_skills/delegate-task/ /path/to/openclaw/skills/
```

### 2. Register MCP server with env vars

openclaw uses [mcporter](https://github.com/steipete/mcporter) as its MCP runtime. Register the server and pass env vars in one command:

```bash
mcporter config add scion --command "scion-mcp" \
  --env SCION_HOST_SKILL_DIRS=/path/to/openclaw/skills \
  --env SCION_WORKSPACE=/path/to/Scion \
  --env SCION_API_KEY=sk-xxx
```

---

## Environment Variables (Agent-Specific)

The three env vars in each agent's setup above are the most important. For the **full env var list**, config files reference, and advanced settings, see the [Advanced Configuration](../../README.md#-advanced-configuration) in the main README.

<details>
<summary>What needs <code>SCION_API_KEY</code>?</summary>

| Capability | Without API Key | With API Key |
|-----------|----------------|--------------|
| `execute_task` | ✅ works (local skills only) | ✅ + cloud skill search |
| `search_skills` | ✅ works (local results only) | ✅ + cloud results |
| `fix_skill` | ✅ works | ✅ works |
| `upload_skill` | ❌ fails | ✅ uploads to cloud |

All tools default to `"all"` (local + cloud) and **automatically fall back** to local-only if no API key is configured. No need to change tool parameters.

</details>

---

## How It Works

```
Your Agent (nanobot / openclaw / ...)
  │
  │  MCP protocol (stdio)
  ▼
scion-mcp              ← 4 tools exposed
  ├── execute_task           ← multi-step grounding agent loop
  ├── search_skills          ← local + cloud skill search
  ├── fix_skill              ← repair a broken SKILL.md
  └── upload_skill           ← push skill to cloud community
```

The two host skills teach the agent **when and how** to call these tools:

| Skill | MCP Tools | Purpose |
|-------|-----------|---------|
| **skill-discovery** | `search_skills` | Search local + cloud skills → decide: follow it yourself, delegate, or skip |
| **delegate-task** | `execute_task` `search_skills` `fix_skill` `upload_skill` | Delegate tasks, search skills, repair broken skills, upload evolved skills |

Skills auto-evolve inside `execute_task` (**FIX** / **DERIVED** / **CAPTURED**). After every call, your agent reports results to the user via its messaging tool.

> [!NOTE]
> For full parameter tables, examples, and decision trees, see each skill's SKILL.md directly.