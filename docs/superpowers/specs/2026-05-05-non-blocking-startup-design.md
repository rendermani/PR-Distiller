# Non-blocking startup, embedding-download progress, and the Encrypted Vault

**Status:** Draft (M1)
**Date:** 2026-05-05
**Milestone:** 1 of 2 (M2 will cover dynamic Ollama model pulls + memory pre-check)

## Goal

Make `make up` produce a UI that opens immediately, with live, byte-level
progress on the embedding-model download persisted across container restarts,
plus a single dedicated page for managing all user-supplied credentials.

## Problem

Today the FastAPI process constructs `LightRAGManager()` at module import time
in `api.py`. That construction triggers a synchronous download of the
sentence-transformers embedding model (default ~440 MB) into a non-persistent
location (`/root/.cache/huggingface/`), which means:

- The HTTP server doesn't open its port until the model finishes downloading
  and loads into memory — minutes on a fresh install.
- Every container recreate (`docker compose down -v`, image rebuild, host
  reboot) wipes the cache, so first-start cost recurs.
- HuggingFace gated/rate-limited models silently fail because no
  `HUGGINGFACE_HUB_TOKEN` plumbing exists.
- The UI has no way to show progress; users see a frozen dashboard or a
  connection-refused error.
- A user who clicks "Run Pipeline" while the model is still loading gets a
  500 deep inside ChromaDB instead of a clean "wait" state.
- All credentials live in a single Settings modal that mixes provider-LLM
  keys with operator settings; trustworthy presentation matters for adoption.

## Non-goals (M2 territory)

- Dynamic Ollama model pulls from the UI with live progress streaming.
- Memory pre-flight check ("this 30 B model won't fit in your VRAM").
- Auto-rotating webhook secrets without operator intervention.
- HF token live validation endpoint.

## Decisions

| Question | Decision |
|---|---|
| M1 vs one-shot delivery | M1 covers everything except dynamic Ollama pull + memory check |
| Progress granularity | Byte-level (`bytes_downloaded / bytes_total`) via `tqdm` interception |
| Download trigger | Eager at backend boot, in a background thread |
| Status transport | Server-Sent Events stream (primary) + 10 s polling backup + initial snapshot fetch on mount |
| HF cache location | New dedicated `hf_cache` volume mounted at `/cache/huggingface`; user data stays in the existing `backend_data` volume |
| HF token storage | Fernet-encrypted in `config.json` (`huggingface_token_enc`), env var `HUGGINGFACE_HUB_TOKEN` wins when set |
| Credentials surface | New page at `/secrets`, "🔒 Encrypted Vault — AES-256" |
| Webhook UX | Subsection on the Vault page with copy-URL and one-shot reveal-and-copy of the secret |
| Env-overridden fields | Inputs disabled, masked value, `from environment` chip, tooltip explains how to change |
| Job queueing | `trigger_job` enqueues into `system_status.queued_jobs` when embedding state ≠ ready; auto-drains on ready transition |

## Architecture overview

```
                                ┌─────────────────────────────┐
                                │ Browser (page.tsx)          │
                                │  • on mount: GET /status    │
                                │  • EventSource /events       │
                                │  • setInterval 10 s GET      │
                                │  • render banner + queue    │
                                └──────────────┬──────────────┘
                                               │
                                               ▼
                  Next.js proxy ── attaches Bearer token, streams SSE through
                                               │
                                               ▼
   ┌───────────────────────────────────────────────────────────────────────┐
   │  FastAPI (backend/api.py)                                             │
   │                                                                       │
   │  GET /api/system/status   ── snapshot of SystemStatus singleton       │
   │  GET /api/system/events   ── SSE stream of mutations                  │
   │  POST /api/system/embedding/retry   (auth)                            │
   │                                                                       │
   │  startup hook (lifespan):                                             │
   │    spawn background thread → install tqdm hook → load embedding      │
   │    model → on success: db_proxy.bind(real_db); status.update("ready")│
   │                                                                       │
   │  trigger_job: if status.is_ready(): run; else: status.enqueue(...)    │
   │  status.on_ready_callback: drains queue via run_coroutine_threadsafe │
   └───────────────────────────────────────────────────────────────────────┘
                                  │                   │
                                  ▼                   ▼
                       SystemStatus singleton    Lazy db proxy
                       (lock, subscribers,       (blocks attribute access
                        snapshot, queue)          until real db is bound)
```

## Storage & startup (Section 1)

### Volumes

`docker-compose.yml` gains a new top-level volume:

```yaml
volumes:
  backend_data:
  hf_cache:
```

The backend service mounts both:

```yaml
volumes:
  - backend_data:/app/data
  - hf_cache:/cache/huggingface
```

Backend env additions:

```yaml
environment:
  - HF_HOME=/cache/huggingface
  - SENTENCE_TRANSFORMERS_HOME=/cache/huggingface/sentence_transformers
  - HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN:-}
```

The web-ui service does **not** mount `hf_cache`. (It only needs the
`backend_data` volume, read-only, for the auth token file.)

### Env-override detection

`backend/system_status.py::env_overrides()` returns a dict like
`{"github_token": False, "huggingface_token": True, "provider_api_keys.openai": False, ...}`,
computed once at startup by checking which canonical env vars are non-empty.
Mapped env vars:

| Field | Env var |
|---|---|
| `github_token` | `GITHUB_TOKEN` |
| `huggingface_token` | `HUGGINGFACE_HUB_TOKEN` |
| `provider_api_keys.openai` | `OPENAI_API_KEY` |
| `provider_api_keys.anthropic` | `ANTHROPIC_API_KEY` |
| `provider_api_keys.google` | `GOOGLE_API_KEY` |
| `provider_api_keys.openrouter` | `OPENROUTER_API_KEY` |
| `github_webhook_secret` | `GITHUB_WEBHOOK_SECRET` |

The dict is exposed via `GET /api/config` as a new field
`env_overrides`. The UI reads it to decide which inputs to disable.

When env is set, the encrypted-config value is **ignored** (env wins), and
the input is rendered read-only. When env is unset, the encrypted value is
the source of truth and the input is editable.

### Lazy `db` proxy

`backend/db_proxy.py` defines a `_LazyDbProxy` class with a
`threading.Event` and a `__getattr__` that blocks until the real
`LightRAGManager` is bound via `bind(real)`. A class attribute
`_NON_BLOCKING_METHODS = {}` is reserved for future fast-path attributes
(none in M1).

`api.py` constructs `db = _LazyDbProxy()` at module import time. The FastAPI
`lifespan` startup hook spawns a background thread; the thread:

1. Calls `system_status.update(state="downloading", model_name=...)`.
2. Installs the `tqdm` hook (see Section 4 below).
3. Constructs the real `LightRAGManager()` — this is where downloads happen.
4. On success: `db.bind(real_manager)`, `system_status.update(state="ready")`.
5. On exception: `system_status.update(state="error", error=str(e))`.
6. Uninstalls the `tqdm` hook in a `finally`.

`/api/health`, `/api/system/status`, and `/api/system/events` never touch
`db`, so they answer immediately. Every other endpoint that uses `db`
blocks transparently until ready.

## Status API + SSE (Section 2)

### `SystemStatus` singleton

`backend/system_status.py`:

```python
class SystemStatus:
    embedding: dict      # {state, model_name, bytes_downloaded, bytes_total, error?}
    queued_jobs: list    # [{job_id, repo, queued_at, reason}, ...]
    _lock: threading.Lock
    _subscribers: list[asyncio.Queue]
    _on_ready: Callable | None  # set by JobOrchestrator at init time
```

States: `idle | downloading | loading | ready | error`. `loading` is the
post-download phase where weights initialize in memory (no byte progress
during this step; UI shows a pulse animation).

Methods:

- `snapshot() -> dict` — read-only deep copy.
- `update(**kwargs) -> None` — mutates state under lock; if `state` flips
  to `ready` and `_on_ready` is set, schedule the callback on the FastAPI
  event loop via `run_coroutine_threadsafe`.
- `subscribe() -> asyncio.Queue` / `unsubscribe(q)` — for SSE handlers.
- `enqueue_job(...)`, `drain_queue()` — for the orchestrator.
- `update_download_progress(bytes_downloaded, bytes_total)` — throttled fan-out
  (see "Throttle" below).
- `env_overrides() -> dict` — see Section 1.

### Throttle

`update_download_progress` writes the new values unconditionally, but only
fans out to subscribers when **either** 250 ms have passed since the last
fan-out **or** the percentage has advanced by ≥ 1 %. Otherwise SSE is silent.
This keeps a 4 GB download from generating ~4 000 events/sec.

### Endpoints

`GET /api/system/status` — auth-free; returns `snapshot()` plus
`env_overrides()` flattened in.

`GET /api/system/events` — auth-free; SSE stream. Implementation
sketch:

```python
@app.get("/api/system/events")
async def system_events(request: Request):
    async def event_stream():
        q = system_status.subscribe()
        try:
            yield f"data: {json.dumps(system_status.snapshot())}\n\n"
            while not await request.is_disconnected():
                try:
                    update = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(update)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            system_status.unsubscribe(q)
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
```

`POST /api/system/embedding/retry?clean=false` — auth-gated. Resets
`embedding.state` to `idle`, optionally `shutil.rmtree`s the cache dir,
restarts the background load thread.

### Proxy compatibility

The Next.js proxy already streams the upstream body through. We add a
trivial change: do **not** override `Cache-Control` on responses with
`Content-Type: text/event-stream`. The `X-Accel-Buffering: no` header
flows through naturally and disables nginx-style buffering if the operator
fronts the proxy with one. No timeout is set on the upstream `fetch`, which
is fine for SSE.

## UI integration (Section 3)

### Banner

The existing top-of-page banner pattern (LLM-unreachable; `page.tsx:131`)
gains a sibling for the embedding state. Banner content per state:

- `downloading`: "Downloading embedding model: {model_name} •
  {format_bytes(bytes_downloaded)} / {format_bytes(bytes_total)} • {pct}%"
  with the gradient progress bar moving live. Below the bar, when no HF
  token is configured **and** no env override, a secondary line: *"Tip:
  downloads are throttled without a HuggingFace token. → Set one in the
  Vault."* (link to `/secrets`).
- `loading`: "Loading model weights into memory…" with pulse animation.
- `error`: red, error message verbatim, plus a `Retry` button (POST to
  `/api/system/embedding/retry`). When the error mentions
  authentication or gating, the button text becomes
  `Set HF token & retry` and links to `/secrets`.
- `ready`: banner hides.

### State-update flow

```ts
useEffect(() => {
  apiFetch('/api/system/status').then(r => r.json()).then(setSystemStatus);
  const es = new EventSource('/api/proxy/api/system/events');
  es.onmessage = e => setSystemStatus(JSON.parse(e.data));
  let backoff = 1000;
  es.onerror = () => {
    es.close();
    apiFetch('/api/system/status').then(r => r.json()).then(setSystemStatus);
    setTimeout(reconnect, Math.min(backoff *= 2, 30000));
  };
  const t = setInterval(() => {
    apiFetch('/api/system/status').then(r => r.json()).then(setSystemStatus);
  }, 10_000);
  return () => { es.close(); clearInterval(t); };
}, []);
```

### Run Pipeline button

Stays clickable. When `system_status.embedding.state !== "ready"` the
backend returns `{job_id, status: "queued", reason}`. UI renders the job
card with a greyed-out spinner and the reason text (e.g., "Queued —
waiting for embedding model"). When the embedding state flips to `ready`,
the orchestrator drains the queue: the same `job_id` transitions to
`running`, and the existing job-status polling loop in `page.tsx`
displays it with no UI changes needed.

### Vault page (`/secrets`)

Route: `web-ui/src/app/secrets/page.tsx`. Headed
`🔒 Encrypted Vault — AES-256`. Subtitle: *"Stored encrypted at rest with
Fernet (AES-128 in CBC mode + HMAC-SHA256). Keys you set via environment
variables are read-only here."*

Sections, top to bottom:

1. **GitHub** — `github_token` field with the live validator already in
   the codebase (`/api/github/validate`). Disabled when env-overridden.
2. **HuggingFace** — `huggingface_token` field. Subtitle: *"Optional.
   Without a token, downloads are rate-limited and gated models will fail."*
   No live validator in M1.
3. **LLM Providers** — one row per provider key (`openai`, `anthropic`,
   `google`, `openrouter`). Each row is a password input and a
   per-provider env-override indicator. The currently active provider gets
   a ★ marker.
4. **GitHub Webhook**:
   - **Webhook URL** — read-only. Computed as
     `{public_base}/api/webhooks/github` where `public_base` comes from a
     new `WEBHOOK_PUBLIC_URL` env var; if unset, falls back to
     `request.url.scheme + "://" + request.url.netloc` of the GET, with a
     yellow warning that the dashboard URL might not be the publicly
     reachable backend URL. Copy button.
   - **Webhook secret** — three states (Generate / from environment /
     editable). Reveal-and-copy is one-shot: click reveals and copies to
     clipboard, after 3 s the field re-masks. `Rotate` button regenerates.
   - **Subscribed events hint** — static text: *"In GitHub's webhook
     config, subscribe to: Pull requests"*.

The existing Settings modal keeps its provider-keys block for now (working,
non-breaking). A small footer in the modal links to `/secrets`. Migrating
the modal away from inline credentials is M2.

### Env-overridden inputs

Visual treatment:

```
┌─────────────────────────────────────────────────────┐
│ HuggingFace Token  [from environment]               │
│ ┌─────────────────────────────────────────────────┐ │
│ │ ••••••••••••••••••••••••••••     (read-only)    │ │
│ └─────────────────────────────────────────────────┘ │
│ Set via HUGGINGFACE_HUB_TOKEN. Update your env or   │
│ unset it to edit here.                              │
└─────────────────────────────────────────────────────┘
```

Tooltip on the chip: *"This value is set via the
`HUGGINGFACE_HUB_TOKEN` environment variable."*

## Queueing & progress capture (Section 4)

### `JobOrchestrator` changes

```python
class JobOrchestrator:
    def __init__(self, db, system_status):
        ...
        self._loop = asyncio.get_event_loop()
        system_status.set_on_ready(self._drain_queue)

    def trigger_job(self, payload, current_config):
        job_id = f"job-{int(time.time())}"
        if system_status.is_ready():
            self.active_jobs[job_id] = {"status": "Initializing...", "progress": 0}
            asyncio.create_task(self._execute_distillation(job_id, payload, current_config))
        else:
            self.active_jobs[job_id] = {
                "status": "Queued — waiting for embedding model",
                "progress": 0,
            }
            system_status.enqueue_job(job_id, payload, current_config)
        return job_id

    def _drain_queue(self):
        # Runs on the FastAPI event loop via run_coroutine_threadsafe
        for job_id, payload, config in system_status.drain_queue():
            self.active_jobs[job_id]["status"] = "Starting (was queued)"
            asyncio.create_task(self._execute_distillation(job_id, payload, config))

    def request_cancel(self, job_id):
        if system_status.dequeue_job(job_id):
            self.active_jobs[job_id]["status"] = "Cancelled (was queued)"
            self.active_jobs[job_id]["progress"] = -1
            return True
        # ... existing in-flight cancel logic
```

### Byte-level progress capture

`backend/embedding_loader.py`:

```python
class _ProgressCapturingTqdm(tqdm.auto.tqdm):
    _live: ClassVar[set] = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        type(self)._live.add(self)
        self._push()

    def update(self, n=1):
        result = super().update(n)
        self._push()
        return result

    def close(self):
        type(self)._live.discard(self)
        self._push()
        return super().close()

    @classmethod
    def _push(cls):
        bytes_downloaded = sum(t.n for t in cls._live)
        bytes_total = sum(t.total or 0 for t in cls._live)
        system_status.update_download_progress(bytes_downloaded, bytes_total)


@contextlib.contextmanager
def capture_hf_progress():
    import huggingface_hub.utils
    import sentence_transformers
    originals = (
        huggingface_hub.utils.tqdm.tqdm,
        sentence_transformers.util.tqdm,
    )
    try:
        huggingface_hub.utils.tqdm.tqdm = _ProgressCapturingTqdm
        sentence_transformers.util.tqdm = _ProgressCapturingTqdm
        yield
    finally:
        huggingface_hub.utils.tqdm.tqdm, sentence_transformers.util.tqdm = originals
```

The bg-thread uses `with capture_hf_progress(): real_db = LightRAGManager()`.
The throttle in `update_download_progress` ensures SSE fan-out stays sane.

## Error handling, retry, testing (Section 5)

### Failure modes

| Cause | `error` text | UI treatment |
|---|---|---|
| Network down | `<exception class>: <message>` | Generic Retry button |
| Gated model, no token | `Model is gated. Set HUGGINGFACE_HUB_TOKEN or paste a token in Settings.` | Retry button text becomes "Set HF token & retry", links to `/secrets` |
| Out of disk | `Out of disk space at /cache/huggingface` | Retry button + manual cleanup hint |
| Cache corruption | `<sentence_transformers error>` | Retry with `?clean=true` to wipe cache |

### Tests (sized for M1)

| Layer | Tests |
|---|---|
| `system_status.py` | snapshot/update; subscribe/unsubscribe fan-out; queue enqueue/drain; thread-safety stress (50 concurrent updates); throttle limits SSE rate |
| `embedding_loader.py` | tqdm subclass aggregates correctly across multiple instances; `capture_hf_progress` restores originals on exit/exception |
| `api.py` SSE | TestClient hits `/api/system/events`, reads initial snapshot frame, asserts a `system_status.update()` produces the expected next frame; SSE response carries `text/event-stream` and `X-Accel-Buffering: no` |
| Job queueing | `trigger_job` enqueues when `state="downloading"`; drain runs queued jobs when state flips to `ready`; `request_cancel` removes a queued job |
| Config | `huggingface_token` redacted on GET, sentinel-stripped on POST, encrypted in `huggingface_token_enc`; `env_overrides` reflects which env vars are set |
| Vault page | E2E (manual): mount with no token → input editable; set env, restart, mount → input disabled with "from environment" chip |
| Lazy db proxy | `db.collection.get(...)` blocks until `bind()`; `/api/health` and `/api/system/status` return without blocking |

## Migration notes

- Existing installs: on first start after the upgrade, the
  `hf_cache` volume is empty → embedding model re-downloads once.
  Subsequent restarts hit the cache.
- Existing `huggingface_token_enc` field is added to `config.json` only
  when the user sets a token; absent for unaffected users.
- Job orchestrator API is backwards-compatible: queued jobs use the same
  `job_id`-based status endpoint, no new client work needed for the
  pipeline status loop.
- The Settings modal isn't removed; it still works for users who don't
  navigate to `/secrets`. The footer link nudges them to the new page.

## Out of scope (M2)

- Memory pre-flight check for LLMs (VRAM / unified-memory / RAM
  detection per platform).
- UI-triggered Ollama model pull with streaming progress (would add
  `/api/models/pull` + Ollama `/api/pull` SSE pass-through).
- HF token live validation endpoint.
- Webhook secret auto-rotation on schedule.
- Multi-user vault (would require per-user encrypted scopes; today's
  encryption is process-wide).
