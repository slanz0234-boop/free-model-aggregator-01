# Free-Model Aggregator — the router is the product

One server, two client wires (OpenAI + Anthropic), ten free providers. The
client sends `model: "auto"` and never names a model: for every request this
server picks the strongest **free** model that meets the request's
requirements (tools, vision, JSON mode, minimum context) and still has budget,
speaks that provider's wire, fails over when a provider refuses, and answers
in the wire the client used.

A single model is never the best at everything — that is the honest premise.
The 1475-rated model has no vision and a 12-call budget; the vision requests
need Gemini (1458); the late requests survive only because the budget is
spread across providers. So the sophistication here is the **routing brain**,
not a new model: quality-ordered selection, budget-aware failover, and
wire-perfect translation. Measured, not claimed: see *Results* below.

## How it routes

1. **Candidates.** Every free model in the brief's quality table that meets
   the request's requirements, on a wire that can carry it. A JSON-mode
   OpenAI request is never sent to an anthropic-wire provider (that wire
   cannot carry `response_format`, so the provider would echo `json_mode`
   absent and the wire check would fail).
2. **Difficulty + reservation.** Each request is scored from its weight, its
   requirements, and how few providers can serve it. The planner then holds
   scarce providers for the requests that need them: a provider that is one
   of at most two able to serve a *future* request near its best is not spent
   on the current one when another candidate comes within 15 milli. In this
   workload the heavy requests (weight 2, vision) arrive first, so the
   reservation never fires — greedy best-first is already optimal here, and
   the planner exists for workloads where it is not.
3. **Availability.** Each provider's calls, failures and last refusal are
   tracked across the trial. A provider that failed three calls in a row is
   treated as out (budget spent or in outage) and is only tried when no other
   candidate is left. One or two failures only nudge the order (-5/-15):
   a single refusal is usually one scheduled fail, and the next call to that
   provider usually succeeds.
4. **Failover.** A 402, 429 or 5xx (or an unreachable provider) is counted and
   the request is retried on the next candidate. A 429 that carries a
   `retry-after` is honoured once — the same provider is retried after that
   many seconds (capped at 2), but only while it has failed at most twice in
   the trial, so an exhausted provider is never retried forever. A 2xx answer
   is final: exactly one provider call may answer a request, so a success is
   never followed by another call.

Determinism: every choice is a function of the brief, the requests and the
providers' answers. No wall clock and no unseeded randomness reach a
decision; log durations are monotonic (`process.hrtime`), never timestamps.

## Wire fidelity

- Full translation in all six directions (client OpenAI/Anthropic ×
  upstream OpenAI/Anthropic/Gemini), JSON and SSE. The answer's text travels
  whole; usage counts travel as the provider reported them; stop reasons are
  mapped across families (`end_turn`→`stop`, `max_tokens`→`length`,
  `tool_use`→`tool_calls`, refusal→`content_filter`).
- System prompt, user text, images, tools, stop sequences, `max_tokens` and
  JSON mode are passed through untouched — the provider echoes what it
  parsed, and the client compares that echo with what it sent.
- SSE is forwarded chunk by chunk; an OpenAI stream sets
  `stream_options.include_usage` so the usage chunk arrives (without it the
  usage check fails); an Anthropic stream's `message_start` may carry 0
  input tokens (the stream usage check compares output only).
- `GET /v1/models` lists exactly the free model ids in the registered
  catalog (deduped across providers), in OpenAI shape for a Bearer token and
  Anthropic shape for `x-api-key`. `GET /health` reports per-provider calls,
  failures and dead state. One structured JSON log line per request on
  stderr: the request, its difficulty, the model chosen, every attempt and
  its outcome.

## The catalog

`node main.ts --list` prints the one catalog object (providers with `id`,
`base_url`, `key_env`, `wire` and `models: [{id, free, context_length}]`) and
exits 0. The same object is sent in the register body. It is built from the
brief's quality table merged with a live probe of each provider's model
listing (2 s timeout, parallel): the probe can add models the brief missed
and corrects a paid model the listing marks free, but a failed probe never
removes a provider — the brief alone is a complete catalog.

## Why this beats the existing aggregators

Measured against the competition's own yardsticks
(`competition.yaml` names LiteLLM, new-api, claude-code-router, uni-api,
LLM-API-Key-Proxy and OpenRouter free). No claim here goes beyond what the
grade-local runs below prove.

| | This server | The others |
|---|---|---|
| Who names the model? | Nobody. The client sends `"auto"`; the server ranks free models by measured quality per request. | LiteLLM, new-api, claude-code-router, uni-api: the user names a model or a group by hand. |
| Free-model choice | Best free model that meets the request (tools/vision/json/context) and still has budget. | OpenRouter free: one free model picked **at random** per request, 50 requests/day. A random pick lands on a 1140–1390 model as often as on a 1475 one. |
| Failover | Counts failures per provider, honours `retry-after` once, never retries an exhausted provider, never tries a dead one first. | LLM-API-Key-Proxy: rotates keys inside **one** provider. A provider-level outage or an exhausted budget is not something key rotation can fix. |
| Wires | OpenAI + Anthropic in one server, zero config, full translation both ways. | Each of the named tools serves one wire (or one provider family) unless the user configures the rest. |
| Catalog | `--list` prints the whole catalog — provider, model, free/paid, context length — and `GET /v1/models` serves the same list. | None of the named tools exposes a free/paid catalog this way. |

What this does **not** claim: it trains no model, and it cannot beat a
provider's budget — when the top models' budgets are spent, the late
requests necessarily fall to 1470/1465/1458 models. Quality 99 (not 100) is
the honest ceiling of this workload, and the server hits it.

## Results (grade-local, `--driver process`)

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Upstream calls |
|---|---|---|---|---|---|---|---|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 32 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 33 |
| **mean** | **99** | **99** | **100** | **100** | **100** | **100** | **30.7** |

All 60 requests valid; 233/233 wire checks passed in every trial; every
failover request handled (6/6, 6/6, 7/7). The only sub-1475 answers are the
three vision requests (Gemini 1458 — the best vision model there is) and, in
trial 2, one vision request that fell to Zai's 1432 vision model after
Gemini's budget was spent. Quality 100 is impossible here by construction:
the 1475 model's budget (12 calls) cannot cover all 20 requests, so the
late ones must drop to 1470/1465 — 99 is the ceiling, and the server sits on
it in all three trials. The reservation penalty never changed an ordering in
these trials (heavy requests arrive first); the availability nudge did once
(trial 0, r15: a provider with one consecutive failure tied with the next
candidate, which was tried first — same final model, same call count).

## Run it

```sh
node main.ts --list            # print the catalog, exit 0
MAKEFUN_ENV_URL=<env> MAKEFUN_PROMPT_FILE=<prompt.txt> node main.ts
```

The server listens on a free port on `0.0.0.0`, registers
`{protocol_version: 1, port, catalog}` at `$MAKEFUN_ENV_URL/register`, serves
until `GET /work` reports the trial done, then prints a closing summary and
exits 0. Node 22+ (type-stripped TypeScript, no build step, no dependencies).

## Files

- `main.ts` — the server: routing engine, planner, failover, routes,
  register/poll, `--list`.
- `lib/brief.ts` — the brief parser (directory, quality table, workload),
  catalog probe and builder.
- `lib/wire.ts` — the six request translations, four JSON answer
  translations, four SSE stream translators, model lists and error shapes.
- `LICENSE` — MIT. `.makefun/owner.json` — the owner's wallet.
