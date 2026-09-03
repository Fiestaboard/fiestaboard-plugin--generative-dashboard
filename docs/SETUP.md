# Setup

## 1. Point it at a model

The plugin talks to any OpenAI-compatible chat completions endpoint.

**OpenAI** — get a key from <https://platform.openai.com/api-keys>.

- API Base URL: `https://api.openai.com/v1`
- API Key: your `sk-...` key
- Model: `gpt-4o-mini` is plenty for this job

**Ollama, LM Studio, or anything local** — no cloud, no per-call cost.

- API Base URL: `http://localhost:11434/v1`
- API Key: any non-empty string; local endpoints ignore it
- Model: whatever you have pulled, e.g. `llama3.2`

Small local models are the ones most likely to fail validation. That is safe —
the board falls back to a plain grid of live numbers — but if you see
`degraded: no_llm` persistently, try a larger model.

## 2. Pick what to watch

Open **Watched Variables** and search. The list is grouped by plugin and shows
each variable's current value, so you can recognise what you want.

Good starting points, if you have the plugins enabled:

- `weather.temp_f`, `weather.condition`
- `air_quality.aqi`
- `network_speed.download_mbps`
- `stocks.*` or `currency.*`
- `pihole.blocked_today`

Watch more than fits. Twenty to forty variables gives the model something to
actually choose between; that choosing is the point of the plugin.

## 3. Write notes for the ones that matter

This is the step people skip and then wonder why the ranking feels arbitrary.

A note tells the model what a number *means*:

- `air_quality.aqi` → "over 100 is unhealthy, over 150 keep the windows shut"
- `network_speed.download_mbps` → "we pay for 900, under 400 is a problem"
- `pihole.blocked_today` → "just trivia, only show it on a quiet day"

That last one is worth noticing: a note is also how you tell the model
something is *not* important.

## 4. Pin anything that must always show

**Always Show** offers only variables you already watch. Pinned variables
survive even when nothing about them is interesting. Use it sparingly — every
pin is a slot the model cannot use for whatever is actually happening.

## 5. Tune the sensitivity

`default_threshold_pct` is how far a number must move before the board is worth
redrawing. The default of 5% suits most things.

Override it per variable under **Per-Variable Sensitivity** when a number's
natural noise differs from its significance:

- CPU load swings constantly and rarely matters → `25`
- AQI matters at small moves → `2`

Set it too low and the board redraws constantly and costs money. Set it too
high and it goes stale. If in doubt, start high and lower it.

## 6. Choose a mode

Start with `grid`. Switch to `prose` if you would rather the board tell you what
changed than show you a table.

Prose is stricter with itself: the model may only use numbers exactly as they
were given to it, never computed or rounded. A sentence that invents a figure is
rejected and the board falls back rather than showing you something wrong.

### Do not ask prose mode for percentages

This is a real trap, verified against a local Gemma 4. Put something like
*"always state the percentage change"* in **Extra Prompt Instructions** and the
model will happily oblige:

```
AQI ROSE 441.9354838709677%. KEEP WINDOWS SHUT.
```

That number was computed, not given, so every response is rejected and the
board falls back to the plain grid *permanently* — a dashboard that looks
broken for a reason buried in a settings box.

The guard is doing its job: 441.9354838709677 is both wrong to show on a
split-flap board and a number nobody handed the model. But if you want change
expressed as a percentage, the right move is a per-variable note explaining the
significance ("under 500 is a problem"), not an instruction to do arithmetic.
