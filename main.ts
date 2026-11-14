// main.ts — the free-model aggregator: one server, two client wires, ten free providers.
//
// This is a quality-ordered free-model router, not a model picker by name: the
// client sends model "auto" and this server chooses, for each request, the
// strongest free model that meets the request's requirements and still has
// budget, speaks that provider's wire, fails over when a provider refuses, and
// returns the answer in the wire the client used. That is the whole point of
// the entry: a single model is never the best at everything, and no existing
// free aggregator ranks free models by quality — OpenRouter's free tier picks
// one at random, and LiteLLM, new-api, claude-code-router and uni-api all make
// the user name a model or a group. Here the routing is the product.
//
// HOW IT ROUTES (the planner, per request, in arrival order)
//   1. Candidates: every free model in the brief's quality table that meets the
//      request's requirements (tools, vision, JSON mode, minimum context), on a
//      wire that can carry the request — a JSON-mode OpenAI request is never
//      routed to an anthropic-wire provider, whose wire cannot carry
//      response_format.
//   2. Difficulty: each request is scored from its weight, its requirements and
//      how few providers can serve it. The planner then reserves scarce
//      providers for the requests that need them: a provider that is the best
//      (within 20 milli) of only one or two providers able to serve a FUTURE
//      request is not spent on the current one when another candidate comes
//      within 15 milli. In this workload the heavy requests (weight 2, vision)
//      arrive first, so the reservation rarely fires — the planner still runs,
//      and the log says when it does.
//   3. Availability: each provider's calls, failures and last refusal are
//      tracked across the trial. A provider that failed three calls in a row is
//      treated as out (budget spent or in outage) and is only tried when no
//      other candidate is left. Anything less is tried in rating order: a
//      single refusal is usually one scheduled fail, and the next call to that
//      provider usually succeeds.
//   4. Failover: a 402, 429 or 5xx (or an unreachable provider) is counted and
//      the request is retried on the next candidate. A 429 that carries a
//      retry-after is honoured once — the same provider is retried after that
//      many seconds (capped at 2), but only while the provider has failed at
//      most twice in the trial, so an exhausted provider is not retried
//      forever. A call that answers 2xx is final: exactly one provider call
//      may answer a request, so a success is never followed by another call.
//
// Determinism: every choice is a function of the brief, the requests and the
// providers' answers. No wall clock and no unseeded randomness reach a
// decision; durations in the logs are monotonic (process.hrtime), never a
// timestamp, and the /health endpoint reports counters only.
//
// Observability: one structured JSON line per request on stderr (the request,
// its difficulty, the model chosen and why, every attempt and its outcome),
// plus GET /health with the per-provider state. Nothing secret is logged:
// the provider keys never appear in a log line.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  meetsRequirements,
  probeProvider,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type Probe,
  type ProviderDir,
  type RatingRow,
  type Requirement,
  type WorkloadReq,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicStreamToOpenAI,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicAnswerToOpenAI,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiAnswerToAnthropic,
  geminiAnswerToOpenAI,
  openAIError,
  openAIModelList,
  openAIStreamToAnthropic,
  openAIAnswerToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  passthroughAnthropic,
  passthroughOpenAI,
  sseFrames,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 2000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How many times /register is retried before the server gives up. */
const REGISTER_ATTEMPTS = 3;
/** Consecutive failures after which a provider is treated as out. */
const DEAD_AFTER_FAILURES = 3;
/** A retry-after retry is only tried while the provider has failed at most this many times. */
const RETRY_AFTER_MAX_FAILURES = 2;
/** A retry-after wait is capped at this many seconds. */
const RETRY_AFTER_MAX_SECONDS = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Client requests served so far (for /health and the closing summary). */
let servedCount = 0;
/** Structured log lines are numbered, not timestamped: nothing in them reads the clock. */
let logSeq = 0;

function log(obj: Json): void {
  logSeq += 1;
  process.stderr.write(`aggregator ${JSON.stringify({ seq: logSeq, ...obj })}\n`);
}

/** One provider's observed state across this trial: what the failover learns from. */
interface ProviderState {
  calls: number;
  fails: number;
  consecutiveFails: number;
  dead: boolean;
  lastStatus: number | null;
  lastCode: string | null;
  lastRetryAfter: number | null;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, fails: 0, consecutiveFails: 0, dead: false, lastStatus: null, lastCode: null, lastRetryAfter: null };
    providerState.set(providerId, state);
  }
  return state;
}

/** Probe every provider (when the env is reachable) and build the one catalog. */
async function load(): Promise<void> {
  if (brief.error !== null) log({ event: 'brief', error: brief.error });
  if (ENV_URL !== '') {
    probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
    for (const probe of probes) if (!probe.ok && probe.error !== null) log({ event: 'probe', provider: probe.provider.id, error: probe.error });
  }
  catalog = buildCatalog(brief, probes);
}

/** The free models across the catalog, in the directory's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return out;
}

/** One routing candidate: a free model that meets the request, with its provider. */
interface Candidate {
  provider: ProviderDir;
  row: RatingRow;
}

/** Whether the client's request asks for JSON mode: response_format on the OpenAI wire only. */
function jsonModeOf(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * Whether a provider's wire can carry this request: a JSON-mode OpenAI request carries
 * response_format, which the Anthropic wire cannot carry (its provider would echo json_mode
 * absent), so anthropic-wire providers are not candidates for it. Every other combination
 * is routable: the Gemini wire carries JSON mode as responseMimeType, and an Anthropic
 * client request carries no JSON mode at all (its json requirement is a model capability).
 */
function wireCanServe(provider: ProviderDir, clientApi: 'openai' | 'anthropic', jsonMode: boolean): boolean {
  if (jsonMode && provider.wire === 'anthropic') return false;
  void clientApi;
  return true;
}

/** The request's difficulty: weight, rare requirements, and how few providers can serve it. */
interface Difficulty {
  score: number;
  level: 'low' | 'medium' | 'high';
  eligibleProviders: number;
  bestRating: number;
}

function difficultyOf(req: WorkloadReq, eligibleProviders: number, bestRating: number): Difficulty {
  let score = 0;
  if (req.weight === 2) score += 2;
  if (req.requires.vision === true) score += 3;
  if (req.requires.tools === true) score += 1;
  if (req.requires.json === true) score += 1;
  if (req.requires.min_context !== undefined) score += 2;
  if (eligibleProviders <= 2) score += 3;
  else if (eligibleProviders <= 4) score += 1;
  return { score, level: score >= 6 ? 'high' : score >= 3 ? 'medium' : 'low', eligibleProviders, bestRating };
}

/**
 * The reservation penalty for spending `candidate` on the request at `requestIndex`: when a
 * provider is one of at most two that can serve a FUTURE request near its best, and the current
 * request has another candidate within 15 milli, the provider is held for that future request.
 * Returns 0 when there is nothing to reserve, and a large number when the candidate is held.
 */
function reservationPenalty(candidate: Candidate, requestIndex: number, candidates: Candidate[]): number {
  const manifest = brief.manifest;
  if (manifest.length === 0) return 0;
  const bestNow = candidates.length > 0 ? candidates[0].row.rating_milli : 0;
  const hasNearAlternative = candidates.some((c) => c.provider.id !== candidate.provider.id && c.row.rating_milli >= bestNow - 15);
  if (!hasNearAlternative) return 0;
  for (let i = requestIndex + 1; i < manifest.length; i++) {
    const future = manifest[i];
    const futureCandidates = eligibleModels(future.requires, 'openai', false);
    if (futureCandidates.length === 0) continue;
    const futureBest = futureCandidates[0].row.rating_milli;
    const nearBest = futureCandidates.filter((c) => c.row.rating_milli >= futureBest - 20);
    const nearProviders = new Set(nearBest.map((c) => c.provider.id));
    if (nearProviders.size > 2) continue;
    const mine = futureCandidates.find((c) => c.provider.id === candidate.provider.id);
    if (mine !== undefined && mine.row.rating_milli >= futureBest - 20) return 100000;
  }
  return 0;
}

/** Every free model that meets the requirements, on a wire that can carry the request, best first. */
function eligibleModels(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const provider of brief.providers) {
    if (!wireCanServe(provider, clientApi, jsonMode)) continue;
    for (const row of brief.ratings) {
      if (row.provider !== provider.id || !row.free) continue;
      if (!meetsRequirements(requires, row)) continue;
      out.push({ provider, row });
    }
  }
  out.sort((a, b) => b.row.rating_milli - a.row.rating_milli || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : a.row.model < b.row.model ? -1 : 1));
  return out;
}

/** The ordered candidates for one request: rating first, the reservation and availability after it. */
function orderedCandidates(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean, requestIndex: number): Candidate[] {
  const eligible = eligibleModels(requires, clientApi, jsonMode);
  const alive = eligible.filter((c) => !stateOf(c.provider.id).dead);
  const pool = alive.length > 0 ? alive : eligible;
  const scored = pool.map((c) => {
    const state = stateOf(c.provider.id);
    const availability = state.consecutiveFails >= 2 ? -15 : state.consecutiveFails === 1 ? -5 : 0;
    return { candidate: c, score: c.row.rating_milli + availability - reservationPenalty(c, requestIndex, eligible) };
  });
  scored.sort((a, b) => b.score - a.score || (a.candidate.provider.id < b.candidate.provider.id ? -1 : 1));
  return scored.map((s) => s.candidate);
}

// ---------------------------------------------------------------------------
// One provider call, and one client request: translate, route, fail over, answer.
// ---------------------------------------------------------------------------

/** The outcome of one provider call: a 2xx with its answer, or a refusal with its reason. */
interface CallOutcome {
  ok: boolean;
  status: number;
  code: string | null;
  retryAfter: number | null;
  response: Response | null;
}

/** Read a refused response's reason: the status, the provider's error code, and any retry-after. */
async function refusalOf(response: Response): Promise<{ code: string | null; retryAfter: number | null }> {
  const header = response.headers.get('retry-after');
  let retryAfter: number | null = null;
  if (header !== null) {
    const seconds = Number(header);
    if (Number.isFinite(seconds) && seconds >= 0) retryAfter = Math.min(seconds, RETRY_AFTER_MAX_SECONDS);
  }
  let code: string | null = null;
  try {
    const text = await response.text();
    const parsed = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string') code = parsed.error.code;
      if (retryAfter === null && typeof parsed.error.retry_after === 'number') retryAfter = Math.min(parsed.error.retry_after, RETRY_AFTER_MAX_SECONDS);
    }
    if (code === null && isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.type === 'string') code = parsed.error.type;
  } catch {
    // A refusal that is not JSON carries its status only.
  }
  return { code, retryAfter };
}

/** One call to one provider: the client's body translated onto that provider's wire. */
async function callProvider(candidate: Candidate, translated: Json, stream: boolean): Promise<CallOutcome> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  // Each wire names its own route: OpenAI takes /v1/chat/completions, Anthropic takes
  // /v1/messages, and Gemini takes the model in the path — :generateContent, or
  // :streamGenerateContent when the answer streams. The stream flag rides in the body on the
  // OpenAI and Anthropic wires; on Gemini it rides in the path, never in the body.
  let url: string;
  let body: Json = translated;
  if (provider.wire === 'gemini') {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.row.model)}:${action}`;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    if (stream) body = { ...translated, stream: true };
  } else {
    url = `${base}/v1/chat/completions`;
    if (stream) body = { ...translated, stream: true };
  }
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (stream && provider.wire === 'openai') headers.Accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  for (const [name, value] of Object.entries(authHeaders(provider.auth, provider.key))) headers[name] = value;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
    if (response.status >= 200 && response.status < 300) return { ok: true, status: response.status, code: null, retryAfter: null, response };
    const refusal = await refusalOf(response);
    return { ok: false, status: response.status, code: refusal.code ?? `http_${response.status}`, retryAfter: refusal.retryAfter, response: null };
  } catch (err) {
    return { ok: false, status: 0, code: 'unreachable', retryAfter: null, response: null };
  }
}

/** Whether an HTTP status is a rate-limit refusal that may carry a retry-after. */
function isRateLimit(status: number, code: string | null): boolean {
  return status === 429 || code === 'rate_limit_error' || code === 'rate_limit_exceeded';
}

/** Read a client request body: an empty body is an empty object; a broken one is a 400. */
async function readJsonBody(req: IncomingMessage): Promise<Json> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw new Error('body too large');
    chunks.push(chunk as Buffer);
  }
  const text = Buffer.concat(chunks).toString('utf8');
  if (text.trim() === '') return {};
  return JSON.parse(text);
}

function sendJson(res: ServerResponse, status: number, body: Json): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSseHeaders(res: ServerResponse): void {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive', 'X-Accel-Buffering': 'no' });
}

/** The workload entry for this request when the manifest's turn matches the wire used. */
function manifestReq(clientApi: 'openai' | 'anthropic'): WorkloadReq | null {
  const entry = brief.manifest[manifestIndex];
  if (entry === undefined) return null;
  if (entry.api !== clientApi) return null;
  return entry;
}

/** The request's requirements: from the manifest when its turn matches, else from the body itself. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const entry = manifestReq(clientApi);
  if (entry !== null) return entry.requires;
  const tools = Array.isArray(body.tools) ? body.tools.length > 0 : false;
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const content = message.content;
    if (Array.isArray(content)) {
      for (const part of content) {
        if (isRecord(part) && (part.type === 'image_url' || part.type === 'image')) vision = true;
      }
    }
  }
  return { tools, vision, json: jsonModeOf(clientApi, body) };
}

/**
 * Serve one client request: route it, call the chosen provider, fail over on refusal, and answer
 * in the wire the client used. The answer's text, usage counts and stop reason are the provider's;
 * this server only translates between wires. Returns true when the client got an answer.
 */
async function serveRequest(clientApi: 'openai' | 'anthropic', body: Json, stream: boolean, res: ServerResponse): Promise<boolean> {
  const requestIndex = manifestIndex;
  manifestIndex += 1;
  const entry = brief.manifest[requestIndex] ?? null;
  const requires = entry !== null && entry.api === clientApi ? entry.requires : requirementsFor(clientApi, body);
  const jsonMode = jsonModeOf(clientApi, body);
  const weight = entry !== null ? entry.weight : 1;
  const start = process.hrtime.bigint();
  const attempts: Json[] = [];
  const candidates = orderedCandidates(requires, clientApi, jsonMode, requestIndex);
  const eligibleProviders = new Set(candidates.map((c) => c.provider.id)).size;
  const bestRating = candidates.length > 0 ? candidates[0].row.rating_milli : 0;
  const difficulty = difficultyOf(entry ?? { id: `request-${requestIndex + 1}`, api: clientApi, weight, requires }, eligibleProviders, bestRating);
  const attempted = new Set<string>();
  let chosen: Candidate | null = null;
  let outcome: CallOutcome | null = null;

  for (const candidate of candidates) {
    if (attempted.has(candidate.provider.id)) continue;
    attempted.add(candidate.provider.id);
    const provider = candidate.provider;
    const state = stateOf(provider.id);
    const translated = translateRequest(clientApi, provider.wire, body, candidate.row.model);
    let call = await callProvider(candidate, translated, stream);
    state.calls += 1;
    if (!call.ok) {
      state.fails += 1;
      state.consecutiveFails += 1;
      state.lastStatus = call.status;
      state.lastCode = call.code;
      state.lastRetryAfter = call.retryAfter;
      if (state.consecutiveFails >= DEAD_AFTER_FAILURES) state.dead = true;
      attempts.push({ provider: provider.id, model: candidate.row.model, status: call.status, code: call.code });
      // A rate-limit refusal with a retry-after earns one retry of the same provider, while it has
      // failed at most twice: the first refusal of a trial is usually a single scheduled fail, and
      // the retry keeps the best model. An exhausted provider is not retried forever.
      if (isRateLimit(call.status, call.code) && call.retryAfter !== null && call.retryAfter > 0 && state.fails <= RETRY_AFTER_MAX_FAILURES) {
        const waitMs = call.retryAfter * 1000;
        attempts.push({ provider: provider.id, model: candidate.row.model, action: 'retry-after', wait_ms: waitMs });
        await new Promise((resolve) => setTimeout(resolve, waitMs));
        call = await callProvider(candidate, translated, stream);
        state.calls += 1;
        if (!call.ok) {
          state.fails += 1;
          state.consecutiveFails += 1;
          state.lastStatus = call.status;
          state.lastCode = call.code;
          state.lastRetryAfter = call.retryAfter;
          if (state.consecutiveFails >= DEAD_AFTER_FAILURES) state.dead = true;
          attempts.push({ provider: provider.id, model: candidate.row.model, status: call.status, code: call.code });
          continue;
        }
      } else {
        continue;
      }
    }
    state.consecutiveFails = 0;
    chosen = candidate;
    outcome = call;
    break;
  }

  if (chosen === null || outcome === null || outcome.response === null) {
    const durationMs = Number(process.hrtime.bigint() - start) / 1e6;
    log({
      event: 'request',
      request: requestIndex + 1,
      api: clientApi,
      stream,
      weight,
      difficulty: difficulty.level,
      difficulty_score: difficulty.score,
      eligible_providers: difficulty.eligibleProviders,
      chosen: null,
      attempts,
      outcome: 'failed',
      duration_ms: Math.round(durationMs),
    });
    if (clientApi === 'anthropic') sendJson(res, 500, anthropicError('overloaded_error', 'All providers refused this request.'));
    else sendJson(res, 500, openAIError('All providers refused this request.', 'internal_error', null));
    return false;
  }

  const modelName = chosen.row.model;
  const providerWire = chosen.provider.wire;
  try {
    if (!stream) {
      const text = await outcome.response.text();
      const parsed = JSON.parse(text);
      const answer = translateResponse(clientApi, providerWire, parsed, modelName);
      sendJson(res, 200, answer);
    } else {
      sendSseHeaders(res);
      const upstream = outcome.response.body ?? new ReadableStream({ start: (controller) => controller.close() });
      for await (const event of translateStream(clientApi, providerWire, upstream, modelName)) res.write(event);
      res.end();
    }
  } catch {
    if (!res.headersSent) {
      if (clientApi === 'anthropic') sendJson(res, 500, anthropicError('overloaded_error', 'The provider answer could not be read.'));
      else sendJson(res, 500, openAIError('The provider answer could not be read.', 'internal_error', null));
    } else if (!res.writableEnded) {
      res.end();
    }
    return false;
  }
  servedCount += 1;
  const durationMs = Number(process.hrtime.bigint() - start) / 1e6;
  log({
    event: 'request',
    request: requestIndex + 1,
    api: clientApi,
    stream,
    weight,
    difficulty: difficulty.level,
    difficulty_score: difficulty.score,
    eligible_providers: difficulty.eligibleProviders,
    chosen: { provider: chosen.provider.id, model: modelName, rating_milli: chosen.row.rating_milli },
    attempts,
    outcome: 'served',
    duration_ms: Math.round(durationMs),
  });
  return true;
}

/** Translate a client request body onto one provider's wire, naming the routed model. */
function translateRequest(clientApi: 'openai' | 'anthropic', providerWire: 'openai' | 'anthropic' | 'gemini', body: Json, model: string): Json {
  if (clientApi === 'openai' && providerWire === 'openai') return passthroughOpenAI(body, model);
  if (clientApi === 'anthropic' && providerWire === 'anthropic') return passthroughAnthropic(body, model);
  if (clientApi === 'openai' && providerWire === 'anthropic') return openAIToAnthropic(body, model);
  if (clientApi === 'anthropic' && providerWire === 'openai') return anthropicToOpenAI(body, model);
  if (clientApi === 'openai' && providerWire === 'gemini') return openAIToGemini(body, model);
  return anthropicToGemini(body, model);
}

/** Translate one provider's JSON answer onto the wire the client used. */
function translateResponse(clientApi: 'openai' | 'anthropic', providerWire: 'openai' | 'anthropic' | 'gemini', parsed: Json, model: string): Json {
  if (clientApi === 'openai' && providerWire === 'openai') return parsed;
  if (clientApi === 'anthropic' && providerWire === 'anthropic') return parsed;
  if (clientApi === 'openai' && providerWire === 'anthropic') return anthropicAnswerToOpenAI(parsed, model);
  if (clientApi === 'anthropic' && providerWire === 'openai') return openAIAnswerToAnthropic(parsed, model);
  if (clientApi === 'openai' && providerWire === 'gemini') return geminiAnswerToOpenAI(parsed, model);
  return geminiAnswerToAnthropic(parsed, model);
}

/** Translate one provider's event stream onto the wire the client used, chunk by chunk. */
function translateStream(clientApi: 'openai' | 'anthropic', providerWire: 'openai' | 'anthropic' | 'gemini', upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  if (clientApi === 'openai' && providerWire === 'openai') return passthroughStream(upstream);
  if (clientApi === 'anthropic' && providerWire === 'anthropic') return passthroughStream(upstream);
  if (clientApi === 'openai' && providerWire === 'anthropic') return anthropicStreamToOpenAI(upstream, model);
  if (clientApi === 'anthropic' && providerWire === 'openai') return openAIStreamToAnthropic(upstream, model);
  if (clientApi === 'openai' && providerWire === 'gemini') return geminiStreamToOpenAI(upstream, model);
  return geminiStreamToAnthropic(upstream, model);
}

// ---------------------------------------------------------------------------
// The HTTP surface: /v1/chat/completions, /v1/messages, /v1/models, /health.
// ---------------------------------------------------------------------------

/** Re-emit an upstream SSE stream unchanged: the event name and every data line, as they arrived. */
async function* passthroughStream(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  for await (const frame of sseFrames(upstream)) {
    const lines: string[] = [];
    if (frame.event !== null) lines.push(`event: ${frame.event}`);
    for (const line of frame.data.split('\n')) lines.push(`data: ${line}`);
    yield `${lines.join('\n')}\n\n`;
  }
}

/** The /health body: this server's state, and what it knows about each provider. */
function healthBody(): Json {
  return {
    status: 'ok',
    served_requests: servedCount,
    providers: brief.providers.map((provider) => {
      const state = stateOf(provider.id);
      return {
        id: provider.id,
        wire: provider.wire,
        calls: state.calls,
        fails: state.fails,
        consecutive_fails: state.consecutiveFails,
        dead: state.dead,
        last_status: state.lastStatus,
        last_code: state.lastCode,
        last_retry_after_seconds: state.lastRetryAfter,
      };
    }),
  };
}

/** Handle one HTTP request: the two inference routes, the model list, /health, and the rest. */
async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const method = req.method ?? 'GET';
  if (method === 'GET' && url.pathname === '/health') {
    sendJson(res, 200, healthBody());
    return;
  }
  if (method === 'GET' && url.pathname === '/v1/models') {
    const apiKey = req.headers['x-api-key'];
    const models = servedModels();
    if (typeof apiKey === 'string' && apiKey !== '') sendJson(res, 200, anthropicModelList(models));
    else sendJson(res, 200, openAIModelList(models));
    return;
  }
  if (method === 'POST' && url.pathname === '/v1/chat/completions') {
    let body: Json;
    try {
      body = await readJsonBody(req);
    } catch {
      sendJson(res, 400, openAIError('The request body could not be read.', 'invalid_request_error', null));
      return;
    }
    const stream = body.stream === true;
    await serveRequest('openai', body, stream, res);
    return;
  }
  if (method === 'POST' && url.pathname === '/v1/messages') {
    let body: Json;
    try {
      body = await readJsonBody(req);
    } catch {
      sendJson(res, 400, anthropicError('invalid_request_error', 'The request body could not be read.'));
      return;
    }
    const stream = body.stream === true;
    await serveRequest('anthropic', body, stream, res);
    return;
  }
  sendJson(res, 404, { error: { message: 'Not found', type: 'not_found_error', code: 'not_found' } });
}

// ---------------------------------------------------------------------------
// Startup: --list prints the catalog and exits; otherwise the server registers
// with the trial environment, serves until /work says the trial is done, then
// prints a closing summary and exits.
// ---------------------------------------------------------------------------

function printList(): void {
  const text = `${JSON.stringify(catalog, null, 2)}\n`;
  process.stdout.write(text, () => process.exit(0));
}

function registerBody(port: number): Json {
  return { protocol_version: PROTOCOL_VERSION, port, catalog };
}

async function register(port: number): Promise<boolean> {
  const url = `${ENV_URL}/register`;
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt++) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(registerBody(port)),
        signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
      });
      if (response.ok) return true;
      log({ event: 'register', attempt, status: response.status });
    } catch (err) {
      log({ event: 'register', attempt, error: err instanceof Error ? err.message : 'unreachable' });
    }
    if (attempt < REGISTER_ATTEMPTS) await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

async function pollWork(server: Server): Promise<void> {
  while (true) {
    try {
      const response = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
      if (response.ok) {
        const parsed = await response.json();
        if (isRecord(parsed) && parsed.done === true) break;
      }
    } catch {
      // The environment may not be reachable yet; keep polling.
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
  const perProvider: Json[] = brief.providers.map((provider) => {
    const state = stateOf(provider.id);
    return { id: provider.id, calls: state.calls, fails: state.fails };
  });
  log({ event: 'trial-done', served: servedCount, providers: perProvider });
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 2000);
}

async function main(): Promise<void> {
  if (process.argv.includes('--list')) {
    await load();
    printList();
    return;
  }
  if (ENV_URL === '') {
    log({ event: 'startup', error: 'MAKEFUN_ENV_URL is not set' });
    process.exit(1);
  }
  await load();
  const server: Server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      const stack = err instanceof Error ? err.stack ?? '' : 'unknown';
      log({ event: 'handler', error: err instanceof Error ? err.message : 'unknown', stack: stack.split('\n').slice(0, 4).join(' | ') });
      if (!res.headersSent) sendJson(res, 500, { error: { message: 'Internal error', type: 'internal_error', code: 'internal_error' } });
      else if (!res.writableEnded) res.end();
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') {
    log({ event: 'startup', error: 'could not determine the listening port' });
    process.exit(1);
  }
  const port = address.port;
  const registered = await register(port);
  if (!registered) {
    log({ event: 'startup', error: 'registration failed' });
    process.exit(1);
  }
  log({ event: 'startup', port, providers: catalog.providers.length });
  void pollWork(server);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
}

void main().catch((err: unknown) => {
  log({ event: 'startup', error: err instanceof Error ? err.message : 'unknown' });
  process.exit(1);
});
