# Generative Dashboard — Design

Date: 2026-09-03
Status: Approved design, pending implementation plan

## Summary

A FiestaBoard plugin that composes a full-board dashboard from a curated set
of variables drawn from other plugins, using an LLM to decide what matters
right now. It regenerates only when the underlying numbers actually move, so
the board stays still when the world is quiet.

Two output modes share everything except the final composition step:

- **grid** — a tiled layout of label/value stats, optionally color-accented.
- **prose** — a sentence or two narrating what changed, sized to the board.

## Goals

1. Surface what is *currently interesting* rather than a fixed set of numbers.
2. Never move things on screen without a reason. Stillness is a feature.
3. Never display a number that is not the real number.
4. Fit any board FiestaBoard supports, using its true dimensions.
5. Degrade in legible steps, and never show a blank or broken board.

## Non-goals

- Discovering variables on its own. The user curates a watchlist — though the
  picker makes that a few clicks, not a typing exercise.
- Replacing `POST /pages/ai/generate` (core's one-shot AI page authoring).
  That is human-triggered and writes a saved page; this is a live plugin.
- Charts, sparklines, or trend glyphs. The board charset has no arrows.
- Partial-board widget use. This plugin takes the whole board.

## Context

Relevant machinery that already exists in FiestaBoard core:

| Piece | Location | Why it matters |
| --- | --- | --- |
| Variable catalog | `registry.get_all_variables_with_metadata()` | Names, types, max lengths, groups — powers the settings picker |
| Live values | `registry.fetch_plugin_data(pid, board)` | Per-plugin values, already cached by core |
| Full context | `registry.build_template_context(board)` | **Deliberately not used** — see Reentrancy |
| Exact grid output | `PluginBase.get_formatted_display()` | Returns literal lines; bypasses template word-wrap |
| Board dimensions | `PluginBase.board` (`BoardContext`) | Real rows/cols at render time |
| Per-board caching | `PluginBase.get_data()` | Keyed by device_type, or `note_array:CxR` |
| Picker protocol | `PluginBase.get_options()` + `remote-options` widget | Searchable, grouped, reorderable variable selection |

`fiestaboard-plugin--generative-ai-art` is the structural precedent for the
repo layout, the OpenAI-compatible settings block, and color-marker output.

### Board constraints

Character set is uppercase only: `A-Z`, `0-9`, space, and
`! @ # $ ( ) - + & = ; : ' " % , . / ?`

There are no lowercase letters, no arrows, no `|`, no `*`, no brackets, and no
underscore. Code 62 renders as degree *or* heart depending on hardware, so it
is never emitted. Color tiles are `{red} {orange} {yellow} {green} {blue}
{violet} {white} {black}`; `{filled}` (code 71) is unavailable on the local
API and is not used.

Sizes are not fixed. Flagship is 6x22, Note is 3x15, and note arrays tile to
6x15, 12x15, 6x30 and others. All dimensions come from `self.board`.

## Architecture

### Repo layout

New repository `fiestaboard-plugin--generative-dashboard`, plugin id
`generative_dashboard`, mirroring the art plugin:

```
__init__.py      GenerativeDashboardPlugin (PluginBase)
source.py        composition, gate, LLM client, validators, renderers
manifest.json    settings schema, emitted variables, demo pages
tests/           unit tests
docs/            SETUP.md and board screenshots
README.md
```

### Execution model — refresh-on-demand

`fetch_data()` must never block. Core gives every plugin a **15 second**
budget during `build_template_context` (`registry.py:1533`) and silently
drops slow ones from that render. LLM calls routinely exceed that.

```
fetch_data(board)
  |
  +- read watchlist values (fast; core-cached)
  +- run the significance gate against the previous snapshot
  +- if material change AND refresh_seconds elapsed AND none in flight:
  |     spawn a one-shot daemon worker -> LLM -> validate -> atomic swap
  +- return the current board immediately
```

Consequences, all intended:

- The render path never waits on the network.
- Generation only advances while the board is actually being displayed, so a
  dashboard that is not in the rotation costs nothing in API calls.
- There is no daemon loop to supervise; each worker exits after one attempt.

### State

Kept per board size, keyed exactly as core keys its cache — `device_type`, or
`note_array:{cols}x{rows}` — because a Note and a Flagship showing this plugin
need different compositions.

Per key: `last_values`, `previous_values`, `board_lines`, `generated_at`,
`degraded_level`, `reason`, `headline`.

Process-wide: an `_inflight` set of board keys under a lock, a `_stopped` flag
set by `cleanup()`, and a `_config_generation` counter bumped by
`on_config_change()` so a worker started under old settings discards its
result instead of swapping it in.

### Reentrancy

`build_template_context` fans out to a `ThreadPoolExecutor` and calls
`get_data()` on **every enabled plugin, including this one**. Calling it from
inside `fetch_data()` would recurse, on a different thread — so a
`threading.local()` guard would not even be visible where the recursion
happens.

This is avoided by construction rather than guarded:

- Collect the distinct plugin ids named in the watchlist.
- Remove `generative_dashboard` from that set unconditionally, so a user
  cannot induce recursion by watching this plugin's own variables.
- Call `registry.fetch_plugin_data(pid, board)` per id.

No nested pool, no recursion, and unrelated plugins are not dragged through a
fetch they did not need.

### The significance gate

Runs on every `fetch_data`, comparing current values to `last_values`:

| Value kind | Material when |
| --- | --- |
| Numeric | `abs(new - old) / max(abs(old), 1e-9) * 100 >= threshold_pct` |
| Anything else | `new != old` |
| Availability | present becomes missing, or missing becomes present |

`threshold_pct` is `thresholds[variable]` if present, otherwise the global
`default_threshold_pct`. Regeneration is additionally forced when there is no
board yet for this size, when config changed, or when the board size changed.

If nothing is material, no LLM call is made and the board is left exactly as
it is. This is the primary anti-jitter mechanism and the primary cost control.

### Board sizing

Computed from the real board, never assumed:

```
tile_columns K = max(1, cols // 11)
tile_width   W = cols // K
tile_budget    = K * rows          # stated to the model, banner-free
                                  # a banner costs one row, i.e. K tiles
prose_budget   = floor(rows * cols * 0.85)   # validated by actually wrapping
```

| Board | Grid | Prose budget |
| --- | --- | --- |
| Flagship 6x22 | 2 columns, up to 12 tiles | ~112 chars |
| Note 3x15 | 1 column, 3 tiles | ~38 chars |
| Note array 12x15 | 1 column, 12 tiles | ~153 chars |
| Note array 6x30 | 2 columns, 12 tiles | ~153 chars |

These numbers are stated to the model explicitly rather than left to inference.

## LLM contracts

Both modes receive: board dimensions and budget, the watchlist with labels and
notes, **current and previous values**, the previously displayed board, and
`extra_instructions`. Temperature defaults to 0.3 — this is layout, not art.

Previous values are what let the model justify emphasis ("this spiked") in
grid mode and narrate change in prose mode. They are free: the gate already
keeps them.

### Grid mode

The model returns tiles, not spacing. It never types a value.

```json
{
  "tiles": [
    {"label": "AQI",  "variable": "air_quality.aqi", "color": "red"},
    {"label": "TEMP", "variable": "weather.temp_f"}
  ],
  "banner": "AIR QUALITY UNHEALTHY",
  "headline": "AQI 168",
  "reason": "AQI rose 31 to 168, crossed unhealthy"
}
```

The plugin substitutes each value from data it already fetched and performs
all character placement. Therefore, by construction: numbers are always real,
width is never exceeded, the charset is never violated, and column alignment
is byte-identical between cycles.

The model still owns every editorial decision — which stats appear, their
order, what they are called, what gets colored, and whether a banner is
warranted.

Validation:

- `tiles` is a list; entries whose `variable` is not in the watchlist are dropped.
- Truncate to `tile_budget`, dropping from the end.
- Any variable in `pinned` missing from the result is appended; if that
  exceeds budget, the lowest-ranked unpinned tile is evicted to make room.
- `label` is uppercased, charset-filtered, and truncated to fit `W` alongside
  its value.
- `color` must be one of the eight; ignored entirely when `use_color` is false.
- `banner` is optional, charset-filtered, and truncated to `cols`. The model
  is told `tile_budget` as if no banner existed, and told that adding one
  costs `K` tiles. If it returns a banner *and* a full tile list, the
  validator drops the lowest-ranked unpinned tiles until
  `len(tiles) <= K * (rows - 1)`. On a 3x15 Note a banner therefore costs a
  third of the board, and the prompt says so.

### Prose mode

```json
{
  "text": "AQI JUMPED FROM 31 TO 168 THIS AFTERNOON. EVERYTHING ELSE IS QUIET - TEMP 61F, CPU STEADY AT 42%.",
  "headline": "AQI 168",
  "reason": "AQI rose 31 to 168"
}
```

Prose cannot dodge hallucinated numbers the way grid mode does, because the
numbers are inside the sentence. The rule is therefore explicit in the prompt:
**use the supplied values exactly as given; do not compute, round, or
abbreviate.** Validation is then an exact match rather than fuzzy
normalization:

1. Extract every numeric token from `text`.
2. Each must appear verbatim in the supplied current-or-previous value set.
3. Any token that does not causes rejection.

Strict, but the instruction sets the model up to satisfy it, and a stilted
sentence is better than a confidently wrong one. Because previous values are
supplied, `FROM 31 TO 168` validates naturally.

Also validated: charset, and that the text wraps into `rows` lines at `cols`
columns. The wrapped block is vertically centered.

### Retry and rejection

A response failing validation is retried once with a stricter, error-specific
instruction. A second failure drops to the failure ladder for that cycle; the
previously displayed board remains up in the meantime.

## Failure ladder

Levels are numbered by how much of the dashboard survives, 0 being intact.
They are *evaluated* in the opposite order — unconfigured first, then whether
any data arrived, then whether the model answered — so the first true
condition wins:

| Level | Condition | Board shows |
| --- | --- | --- |
| 0 | Healthy | LLM-composed grid or prose |
| 1 | LLM unreachable or twice-rejected, data fine | Plugin's own deterministic grid: pinned first, then watchlist order |
| 2 | No watchlist values reachable at all | Humor plus one honest line |
| 3 | Watchlist empty or unconfigured | Actionable setup message, not a joke |

Individual values going missing is **not** a rung — it is orthogonal, and can
happen at any level. Those tiles are dropped and the rest renders normally,
with `degraded` reporting `partial_data`. Level 2 is reached only when *every*
watched value is unreachable.

Level 1 is deterministic and needs no model, which is why it doubles as the
cold-start renderer: the board is useful the instant the plugin is enabled,
before any API call has returned. Prose mode also falls back to this grid,
since a factual list needs no model to produce.

Level 2, on a Flagship:

```
   THE HAMSTERS HAVE
   STOPPED RUNNING ON
   THE WHEEL

   LAST GOOD DATA 14:32
```

Drawn from a small pool — the hamsters, someone unplugged the router, the
numbers are taking a nap, lost contact with the outside world — all uppercase,
charset-clean, and authored to fit 15 columns as well as 22. Gentle, never at
the user's expense.

The `LAST GOOD DATA` line is not decoration. The joke says "you can ignore
this"; the timestamp says how long it has been broken, so a two-day outage is
never mistaken for a two-minute one.

Levels 2 and 3 render through the same wrap-and-center path prose mode uses.

## Settings schema

| Setting | Type | Default | Notes |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | |
| `api_base_url` | string | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `api_key` | string (secret) | — | Required |
| `model` | string | `gpt-4o-mini` | |
| `temperature` | number | `0.3` | 0–2 |
| `output_mode` | enum | `grid` | `grid` or `prose` |
| `refresh_seconds` | int | `300` | Minimum 120; floor between generations |
| `default_threshold_pct` | number | `5` | Global significance threshold |
| `use_color` | bool | `true` | Disables color accents when false |
| `watchlist` | array of string | `[]` | The candidate pool. Picker-driven. Max 100 |
| `labels` | object | `{}` | Display name per variable; written by the picker |
| `pinned` | array of string | `[]` | Subset of `watchlist` that must always appear |
| `notes` | object | `{}` | Per-variable meaning, handed to the model |
| `thresholds` | object | `{}` | Per-variable significance override |
| `extra_instructions` | string | `""` | Appended to the system prompt |
| `custom_system_prompt` | string | `""` | Full override, for parity with the art plugin |

The watchlist is a **candidate pool, not a display list.** Watching 60
variables on a Flagship is legitimate and expected — only 12 tiles fit, and
choosing among the candidates is the model's entire job. Board space is
therefore never the reason to keep the list short.

The real cost is prompt size: each entry contributes its name, label, note,
and current *and* previous value, roughly 40 tokens. A 100-entry watchlist is
about 4k tokens per call — comfortable for a hosted model, heavy for a small
local one on a Pi. 100 is the hard cap; past that `validate_config` rejects.
The feedback below that is honest and self-evident: latency and cost climb,
and the board says how many stats it actually placed via `stat_count`.

### Variable picker

This is the part that decides whether the plugin feels valuable. Choosing
variables by typing `air_quality.aqi` into a text box would be unusable, so
the plugin implements `get_options()` against core's `remote-options` widget
(`manifest.py:47`), which supports exactly what is needed:

```json
"watchlist": {
  "type": "array",
  "items": {"type": "string"},
  "title": "Watched Variables",
  "ui:widget": "remote-options",
  "ui:options": {
    "options_id": "variables",
    "multiple": true,
    "searchable": true,
    "server_search": true,
    "reorderable": true,
    "labels_field": "labels",
    "cache_seconds": 30,
    "placeholder": "Search variables from your enabled plugins"
  }
}
```

`get_options()` builds the catalog from
`registry.get_all_variables_with_metadata()` — the same registry access the
plugin already needs — and returns one `Option` per variable:

| Field | Content |
| --- | --- |
| `value` | `plugin_id.var_name`, the scalar stored in config |
| `label` | The variable's human name |
| `description` | Its description from the source plugin's manifest |
| `group` | The owning plugin's display name, so the list reads grouped |
| `preview` | **Its current value**, fetched live |
| `disabled` | Set, with a reason, for variables that cannot be dashboarded |

`preview` is what makes this pleasant rather than clerical: the picker shows
`Temperature — 61F` instead of an opaque identifier, so the user selects by
recognising the number they already care about.

`disabled` closes two footguns visibly rather than silently:

- Any `generative_dashboard.*` variable, which would otherwise invite the
  recursion the architecture avoids. Shown, greyed, reason given.
- Variables whose `max_length` exceeds a single tile width — the art plugin's
  512-character `art` blob is the motivating case. Nothing sensible can be
  done with one on a dashboard.

`server_search` means `request.query` is filtered plugin-side, so the list
stays responsive as the catalog grows. `reorderable` gives the user an
explicit priority order, which the model receives as a tiebreak hint.

`labels_field: "labels"` lets the widget write display names into the sibling
`labels` object keyed by variable, so renaming `air_quality.aqi` to `AQI`
needs no separate form.

`pinned` is a second `remote-options` field with
`depends_on: ["watchlist"]`, so it only ever offers variables already being
watched.

`notes` and `thresholds` stay keyed-object fields for power users. `notes` is
the one that earns its place — free text like *"over 100 is unhealthy; over
150 keep windows shut"* is how domain meaning enters the system, and it is
what turns the model's ranking from a guess into a judgement. It is also the
extension point for data beyond FiestaBoard's own plugins.

`get_options()` runs on a throwaway instance with draft config applied, on
every keystroke, whether or not the plugin is enabled (`base.py:669`). It must
therefore never require an API key and never call the LLM.

## Emitted variables

| Variable | Type | Notes |
| --- | --- | --- |
| `rows` | array of string | The composed board, one entry per row |
| `prose` | string | Prose text; empty in grid mode |
| `headline` | string | Single most important stat, for use on other pages |
| `reason` | string | Why it last regenerated; also the debugging window |
| `stat_count` | number | Tiles currently displayed |
| `degraded` | string | `""`, `no_llm`, `partial_data`, `no_data`, `unconfigured` |
| `generated_at` | string | ISO timestamp |
| `model` | string | Model that composed it |

`get_formatted_display()` returns the exact lines for standalone page use.

## Testing

Behavior-first, one behavior per test.

- **Gate:** numeric threshold just under and just over; string change; float
  noise below threshold; availability appearing and disappearing; zero and
  near-zero previous values not dividing by zero.
- **Reentrancy:** watchlist naming `generative_dashboard.*` never calls back
  into this plugin; `build_template_context` is never called.
- **Non-blocking:** `fetch_data` returns while a worker is still running;
  concurrent calls spawn only one worker per board key.
- **Grid validation:** unknown variable dropped; over-budget truncated; pinned
  entry reinstated when the model omits it; label truncation at W; color
  suppressed when `use_color` is false; banner reduces tile budget.
- **Prose validation:** a fabricated number is rejected; supplied numbers pass;
  a previous value used in a "from X to Y" sentence passes; overlong text
  rejected; lowercase, `|` and arrow characters rejected.
- **Placement:** exact expected lines at 6x22, 3x15, and 12x15.
- **Failure ladder:** each of the five levels renders its expected board, and
  the `LAST GOOD` timestamp reflects the real last success.
- **Config:** `on_config_change` discards an in-flight worker's result.
- **Picker:** `get_options` returns grouped options with live-value previews;
  `generative_dashboard.*` and over-long variables come back `disabled` with a
  reason rather than omitted; `query` filters server-side; it succeeds with no
  API key configured and never calls the LLM; `pinned` offers only watched
  variables; a watchlist over 100 entries fails `validate_config`.

Per FiestaBoard's testing bar, tests written after their production code must
be proven non-vacuous by breaking the behavior and observing the failure.

## Deferred

- `auto` mode letting the model pick grid or prose per cycle. Tempting, but
  the board flipping between a tile grid and a paragraph is the noisiest thing
  it could do, and it would flip exactly when something changed — the moment
  the board most needs to be readable. Revisit after living with both.
- Trend glyphs and sparklines. Blocked on charset.
- History deeper than one previous snapshot.
- Per-variable custom value formatters.

## Risks

- **Cost.** Mitigated by the gate, by `refresh_seconds`, and by generation only
  advancing while the board is displayed.
- **Small models** may fail validation repeatedly; level 1 keeps the board
  correct while that happens, and `reason` plus `degraded` make it visible.
- **Prompt growth** with watchlist size. Capped at 100 entries (~4k tokens).
  Small local models on a Pi will feel this first; the cap is a guardrail, not
  a recommendation.
