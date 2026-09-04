# Generative Dashboard

A FiestaBoard plugin that builds a full-board dashboard out of variables from
your *other* plugins, and asks an LLM which of them matter right now.

You pick a pool of variables to watch. The plugin watches them, and when
something actually moves it recomposes the board around whatever became
interesting. When nothing moves, it does nothing at all — no API call, no
redraw, no flicker.

## Two output modes

**`grid`** — a tiled layout of label/value stats.

```
CPU     42%  TEMP     61F
AQI    168   BTC   94,120
DISK    71%  UP    4D 2H
```

**`prose`** — a sentence about what changed, sized to the board.

```
   AQI JUMPED FROM 31
   TO 168 THIS
   AFTERNOON. EVERYTHING
   ELSE IS QUIET.
```

Same watchlist, same gating, same failure handling. Only the final composition
step differs.

## The board stays still

Every cycle the plugin reads the watched values and compares them with the
snapshot it last generated from. If nothing moved by more than your sensitivity
threshold, it stops there — the LLM is never called and the board is not
touched.

That single rule does most of the work here. It keeps the board from twitching,
and it means a quiet day costs nothing in API calls.

In `grid` mode the plugin holds the model's *tile choice*, not finished lines,
and substitutes values fresh on every render. So between generations your
numbers stay live while their positions stay put.

## Watch more than fits

The watchlist is a **candidate pool, not a display list.** Watching 60
variables on a Flagship is the intended usage — only 12 tiles fit, and choosing
among the candidates is the whole job. Board space is never the reason to keep
the list short.

The real cost is prompt size: roughly 40 tokens per watched variable. At the
100-variable cap that is about 4k tokens per generation, which is comfortable
for a hosted model and heavy for a small local one on a Pi.

## Picking variables

The settings picker is searchable and grouped by source plugin, and shows each
variable's **current value** next to its name — so you select by recognising
the number you already care about, not by typing `air_quality.aqi`.

Drag to reorder; the order is a priority hint to the model. Variables that
cannot be used are shown greyed with a reason rather than hidden — a value too
wide for a tile, this plugin's own variables (a dashboard cannot watch itself),
or a plugin you have not enabled yet. That last one matters: core only
publishes variables for enabled plugins, so without this the picker would
quietly omit most of what your board can do.

**Notes are the setting worth your time.** A note like *"over 100 is unhealthy,
over 150 keep the windows shut"* is what turns the model's ranking from a guess
into a judgement.

## When things break

| Situation | The board shows |
| --- | --- |
| Healthy | The composed dashboard |
| LLM unreachable or rejected twice | The plugin's own plain grid, pinned first — still live numbers |
| Some values missing | Those tiles dropped, the rest renders |
| Nothing reachable at all | A note about the hamsters, and when data was last good |
| Nothing watched yet | What to do about it |

The outage screen always carries a timestamp. The joke tells you it is fine to
ignore; the timestamp tells you how long it has been broken, so a two-day
outage is never mistaken for a two-minute one.

## Settings

| Setting | Default | What it does |
| --- | --- | --- |
| `watchlist` | `[]` | The candidate pool. Max 100 |
| `labels` | `{}` | Short board label per variable |
| `pinned` | `[]` | Variables that must always appear |
| `notes` | `{}` | What a variable means, per variable |
| `thresholds` | `{}` | Per-variable sensitivity override |
| `output_mode` | `grid` | `grid` or `prose` |
| `api_base_url` | OpenAI | Any OpenAI-compatible endpoint |
| `api_key` | — | Required |
| `model` | `gpt-4o-mini` | Any model the endpoint serves |
| `temperature` | `0.3` | Low on purpose. This is layout, not art |
| `refresh_seconds` | `300` | Floor between generations, not a schedule |
| `default_threshold_pct` | `5` | How far a number must move to count |
| `use_color` | `true` | Let the model flag an outlier with a colour tile |
| `extra_instructions` | `""` | Appended to the system prompt |

## Variables it emits

`rows` (the board, read as `rows.0.text`), `prose`, `headline`, `reason`,
`stat_count`, `degraded`, `generated_at`, `model`.

`reason` is the useful one when you are wondering why the board changed — it
carries the model's own explanation.

## Board sizes

Every size FiestaBoard supports. Dimensions come from the board being rendered,
so a Flagship gets two columns of tiles, a Note gets one, and a stacked note
array gets as many rows as it has. The model is told the exact grid and budget
rather than guessing.

## Development

```bash
./scripts/test.sh          # whole suite
./scripts/test.sh tests/test_gate.py
```

Tests run inside the FiestaBoard dev container, which has pytest and `src/` on
the path. Nothing is installed on the host.

## Licence

MIT
