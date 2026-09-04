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
    MARKET UPDATE
SYMBO GOOG PRIC 339.08
CHAN  1.59 TIME  17:46
DATE          09/03/26
```

The block is centred rather than pinned to the top, so a half-full board reads
as a layout rather than a fault. A stat whose value is too long to leave room
for a label — a date, a timestamp — is given a whole row to itself instead of
being labelled `D`.

**`prose`** — a sentence about what changed, sized to the board.

```
   AQI JUMPED FROM 31
   TO 168 THIS
   AFTERNOON. EVERYTHING
   ELSE IS QUIET.
```

Same watchlist, same gating, same failure handling. Only the final composition
step differs.

## Controlling how often it fires

Two settings decide how busy the board is, and neither is a polling interval —
nothing happens at all unless a value moved.

- **How Much a Number Must Move (%)** — the size of change worth redrawing for.
  Higher is stiller.
- **Minimum Time Between AI Updates** — the ceiling on how often a redraw can
  happen. 300 is five minutes; 1800 gives a calm, cheap board.

A quiet day costs nothing either way: no change, no call.

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

## There is nothing to configure

Out of the box it watches **every variable your enabled plugins expose**. No
picking, no list to curate. Choosing what matters is the model's entire job;
making you pre-choose would just be doing that job twice, worse.

On a board with fourteen plugins enabled that is 139 of 144 variables, roughly
5k tokens per generation. The only ones left out are values wider than the
board itself, which could not be shown whatever the model decided. Reading them
costs about 4ms in steady state, because core already caches every plugin's
data — only the first render after a restart pays to fill that cache.

Set a watchlist only if you want to **narrow** it: mute a noisy variable, or
keep the board to one subject. The picker is searchable, names the owning
plugin in every row, and shows each variable's current value in its tooltip.
Variables that cannot be used appear greyed with the reason — too wide for a
tile, this plugin's own output (a dashboard cannot watch itself), or a plugin
you have not enabled yet.

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
| `watchlist` | `[]` | Empty sends everything. Set it only to restrict |
| `labels` | `{}` | Short board label per variable |
| `pinned` | `[]` | Variables that must always appear |
| `notes` | `{}` | What a variable means, per variable |
| `thresholds` | `{}` | Per-variable sensitivity override |
| `output_mode` | `grid` | `grid` or `prose` |
| `api_base_url` | OpenAI | Any OpenAI-compatible endpoint |
| `api_key` | — | Required |
| `model` | `gpt-4o-mini` | Any model the endpoint serves |
| `temperature` | `0.3` | Low on purpose. This is layout, not art |
| `refresh_seconds` | `300` | Ceiling on how often the AI may redraw |
| `default_threshold_pct` | `5` | How far a number must move to be worth a redraw |
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
