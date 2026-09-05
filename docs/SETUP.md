# Generative Dashboard Setup Guide

Get an AI-curated board running, then tune it — the tuning is where the quality comes from.

## Overview

**What it does:** composes your board from every variable your enabled
plugins expose, choosing what matters right now for the people watching.

**Prerequisites:**
- FiestaBoard ≥ 2.10.0
- An OpenAI-compatible chat endpoint (hosted or local) and its API key
- At least a few data plugins enabled — the dashboard can only show what
  your plugins provide

**Set expectations before you start:** this plugin is a collaboration
between four variables — *what you tell it about yourself, which plugins
feed it, when you want to see what, and which model composes*. All four
affect output, and each is worth a pass. An untuned install works; a tuned
one is the reason to run it.

## Quick Setup

1. **Enable** the plugin under Integrations.
2. **Configure the model.** Endpoint URL, API key, model name. See *Choosing
   a model* below — this choice matters more than any prompt setting.
3. **Write "Who's Watching."** One paragraph: who glances at this wall, your
   schedules, what you care about at which times of day, what bores you.
   This is the single highest-leverage field in the plugin.
4. **Create the page.** Use the demo page, or a template of six
   `{{generative_dashboard.rows.N.text}}` lines, and add it to your rotation.
5. **View, then tune.** Live with it a day, then adjust (see below).

### Connecting an AI model

Any endpoint that speaks the OpenAI chat-completions API works. Fill three
fields:

| Provider | `api_base_url` | `model` (example) | `api_key` |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | your `sk-...` key |
| OpenRouter | `https://openrouter.ai/api/v1` | `google/gemini-3.7-flash` | your `sk-or-...` key |
| Ollama (local) | `http://YOUR-HOST:11434/v1` | `llama3.2` | any non-empty string |
| LM Studio / MLX (local) | `http://YOUR-HOST:8080/v1` | as served | any non-empty string |

Notes:

- For a **local model on another machine**, use that machine's LAN address
  and make sure the server listens on `0.0.0.0`, not just localhost. If
  FiestaBoard runs in Docker on the same machine as the model, use
  `http://host.docker.internal:PORT/v1`.
- Local endpoints usually ignore the key, but the field is required — any
  non-empty value works.
- Nothing but the composition prompt is sent to the endpoint you configure;
  there is no other network destination.

### Choosing a model

The author's experience so far:

- **Hosted mid-tier models (Gemini Flash via OpenRouter) perform well** —
  good theme choices, correct JSON, sensible color banding, and cheap enough
  to re-compose four times an hour without thinking about it.
- **Small local models (e.g. quantized ~26B via Ollama/MLX) have been less
  successful out of the box** — more malformed JSON (the plugin repairs and
  retries, but each miss costs a cycle), tersier prose, and weaker layout
  judgment. This is probably fixable with the right tuning — a lower
  temperature, a trimmed watchlist so the prompt is smaller, and blunter
  `extra_instructions` all help — and the plugin's validation means a weak
  model degrades to a plain-but-correct board, never a wrong one.

If a board feels dumb, try a stronger model *before* touching anything else;
every guarantee is model-independent, but taste is not.

### Tuning, in order of leverage

1. **The audience brief** — rhythms ("one of us commutes Tue–Thu"), stakes
   ("windows open unless AQI says otherwise"), dayparts ("mornings: weather
   and the stock price"), and dismissals ("we don't care about sports").
2. **The model** — see above.
3. **Which plugins are enabled** — the dashboard's palette. Enable what you
   want it to draw from; disable what you never want to see.
4. **Sensitivity and cadence** — `default_threshold_pct` (how big a move
   earns a re-layout) and `refresh_seconds` (the re-layout floor; values
   stay live regardless). Paid endpoint? These are your cost knobs.
5. **Per-variable notes** — for a stat the model keeps misjudging: "over 100
   is unhealthy", "just trivia, quiet days only".

## Template Variables

| Variable | Description |
|---|---|
| `generative_dashboard.rows.{n}.text` | Composed board, one row per entry |
| `generative_dashboard.prose` | Sentence template, prose/auto mode |
| `generative_dashboard.headline` | Most important stat right now |
| `generative_dashboard.reason` | Why the board last changed |
| `generative_dashboard.degraded` | Empty when healthy |
| `generative_dashboard.generated_at` | Last composition time (local) |
| `generative_dashboard.model` | Composing model |
| `generative_dashboard.stat_count` | Tiles currently placed |

## Configuration Reference

| Setting | Default | Notes |
|---|---|---|
| `audience` | — | The editor's brief; leads every prompt |
| `api_base_url` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `api_key` | — | Required |
| `model` | `gpt-4o-mini` | See *Choosing a model* |
| `output_mode` | `auto` | `auto` / `grid` / `prose` |
| `temperature` | `0.3` | Raise only if boards feel repetitive |
| `refresh_seconds` | `900` | Re-layout floor (≥120); values live regardless |
| `default_threshold_pct` | `5` | Move size that earns a re-layout |
| `use_color` | `true` | Range-rule status lights |
| `watchlist` | empty | Empty = everything; set to restrict |
| `pinned` | `[]` | Always-shown variables |
| `notes` | `[]` | Per-variable meaning, `{variable, note}` rows |
| `thresholds` | `[]` | Per-variable sensitivity, `{variable, percent}` rows |
| `extra_instructions` | — | Appended to the system prompt |

Environment variables: `GENERATIVE_DASHBOARD_ENABLED`,
`GENERATIVE_DASHBOARD_API_KEY`, `GENERATIVE_DASHBOARD_API_BASE_URL`,
`GENERATIVE_DASHBOARD_MODEL`, `GENERATIVE_DASHBOARD_OUTPUT_MODE` (all
optional; settings take precedence).

## Troubleshooting

**The board shows plain stats with no title (`degraded: no_llm`).**
The model is unreachable or its replies keep failing validation. Check the
endpoint and key first; with a local model, see *Choosing a model* — the
fallback board is correct, just uncurated.

**A wall of pollen counts and currency rates.**
The cold-start fallback before the first-ever composition. If it persists,
the model has never successfully composed — same causes as above.

**`COMPOSING...` on the bottom row.**
Normal: the first composition is being drafted. It resolves within a minute.

**Boards re-compose too often / API bill surprises.**
Raise `default_threshold_pct` (5 → 10) and `refresh_seconds` (900 → 1800).
Nothing is ever generated when no watched value moved.

**A stat is missing its unit, or a label is clipped.**
Units are inferred from plugin manifests and re-established each
composition; a clipped label means the model ignored its budget — both
self-correct on the next re-layout. Persistent offenders can be renamed via
the watchlist picker's label box.

**Why did the board change?**
`generative_dashboard.reason` carries the model's own explanation, and the
composition log (`composition_log.jsonl` beside the plugin) records every
composition with the model's reasoning — the file to read when tuning.
