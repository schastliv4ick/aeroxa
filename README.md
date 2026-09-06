**English** · [Русский](README.ru.md)

# Aeroxa Legal RAG (MVP)

Backend that answers legal questions about Russian Federation law for foreign
entrepreneurs. A question in Russian, Chinese or English is matched against a corpus of
Russian federal acts, and an LLM writes the answer **only** from the retrieved articles,
citing each one — or abstains and escalates to a lawyer.

Full specification, including what is deliberately out of scope: [docs/SPEC.md](docs/SPEC.md).

```
question ─► condense ─► BGE-M3 hybrid search ─► in-force filter ─► bge-reranker ─► diversify ─┬─► LLM ─► cited answer
  (any lang)  (if history)  (Qdrant, dense+sparse RRF)                                        └─► abstain + escalate
```

The seed corpus is five federal acts — **3961 chunks over 1209 articles**, each at its
current in-force redaction. Cross-lingual retrieval is verified: 有限责任公司的最低注册资本
是多少？ returns Статья 14 «Уставный капитал общества».

## Quick start

Commands are given for **macOS / Linux** (bash or zsh) and **Windows** (PowerShell). Run
them from the repository root.

### 1. Infrastructure

Same on every platform:

```bash
docker compose up -d qdrant postgres
```

If this fails with `dial tcp: lookup registry-1.docker.io: no such host`, Docker Hub is not
reachable from your network — see [Troubleshooting](#troubleshooting).

### 2. Python environment

Python 3.11 or newer.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Two differences that matter:

- The virtualenv layout differs — macOS and Linux create `.venv/bin/`, Windows creates
  `.venv\Scripts\`. There is no `activate` script at the other path.
- The `--extra-index-url` pins CPU-only PyTorch wheels. Use it on Windows and Linux, where
  the default wheel can pull in CUDA libraries you will not use. On macOS it is simply
  unnecessary — PyTorch has no CUDA build for macOS, so the default PyPI wheel is already
  CPU/MPS-only. Adding the flag there is harmless, just pointless.

If PowerShell refuses to run the activation script, allow it for the current session:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

After activation, `python` refers to the virtualenv on both platforms, so every command
below is identical.

### 3. Configuration

**macOS / Linux**

```bash
cp .env.example .env
```

**Windows (PowerShell)**

```powershell
Copy-Item .env.example .env
```

Then set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`. In production these must point at a
self-hosted OpenAI-compatible endpoint inside the perimeter — see
[Confidentiality](#confidentiality).

### 4. Model cache location — do this before the first run

BGE-M3 and the reranker need **~9 GB** of cache between them, because Hugging Face stores
every published weight format. The default location is on your home drive, which is often
the drive with the least room. If it fills up mid-download you get a confusing failure —
`OSError: Can't load the model for 'BAAI/bge-reranker-v2-m3'`, raised in the middle of a
request rather than at startup.

Point the cache at a disk with space, **in the same shell that runs the commands below**:

**macOS / Linux**

```bash
export HF_HOME=/Volumes/data/hf-cache
```

**Windows (PowerShell)**

```powershell
$env:HF_HOME = "D:\hf-cache"
```

This must be a real environment variable — `.env` is read by this application, not by the
Hugging Face libraries, so putting `HF_HOME` there has no effect. To set it permanently,
add the `export` line to `~/.zshrc`, or on Windows run
`[Environment]::SetEnvironmentVariable('HF_HOME', 'D:\hf-cache', 'User')` and open a new
terminal.

### 5. Build the corpus

```bash
python -m ingest.cli run --act fz-14-ooo
```

First run downloads ~2.3 GB of model weights and then embeds every article on CPU. Expect
tens of minutes for a single act, hours for the full seed corpus. Fetched HTML is cached in
`data/raw/`, so re-runs skip the network.

### 6. Serve

```bash
python -m app.main
```

`http://localhost:8000/docs` for the interactive API.

## Asking a question

The request body contains Cyrillic, so how you send it depends on your shell.

**macOS / Linux** — single quotes keep the JSON intact:

```bash
curl -s localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Каков минимальный размер уставного капитала ООО?"}'
```

**Windows (PowerShell)** — use `Invoke-RestMethod`, and keep `charset=utf-8` so Cyrillic
survives. This is the recommended form on Windows:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body '{"question": "Каков минимальный размер уставного капитала ООО?"}'
```

Do **not** use bare `curl` in Windows PowerShell: it is an alias for `Invoke-WebRequest`,
which does not accept `-s`, `-H` or `-d`.

### Sending the body from a file

If your console mangles Cyrillic, put the body in a UTF-8 file and post that. The file must
have **no BOM** — a byte-order mark makes the JSON unparseable.

**macOS / Linux**

```bash
cat > question.json <<'JSON'
{"question": "Каков минимальный размер уставного капитала ООО?"}
JSON

curl -s localhost:8000/ask -H "Content-Type: application/json" -d @question.json
```

**Windows (PowerShell)**

```powershell
[System.IO.File]::WriteAllText("$PWD\question.json",
  '{"question": "Каков минимальный размер уставного капитала ООО?"}',
  (New-Object System.Text.UTF8Encoding $false))

curl.exe -s localhost:8000/ask -H "Content-Type: application/json" -d "@question.json"
```

Three things differ on Windows and all three are required: `curl.exe` rather than `curl`;
`"@question.json"` **in quotes**, because an unquoted `@` starts PowerShell splatting and
fails with `SplattingNotPermitted`; and `WriteAllText` with `UTF8Encoding $false` rather
than `Set-Content -Encoding utf8`, which in Windows PowerShell 5.1 writes a BOM and
produces `Unexpected UTF-8 BOM` on the server.

```jsonc
{
  "answer": "Минимальный размер уставного капитала общества с ограниченной ответственностью составляет 10 000 рублей (Статья 14 Федерального закона «Об обществах с ограниченной ответственностью») [1].",
  "language": "ru",
  "abstained": false,
  "escalate": false,
  "citations": [
    {"n": 1, "act_title": "Об обществах с ограниченной ответственностью", "article_number": "14",
     "revision_index": 57, "revision_date": "2026-08-04", "url": "…", "score": 0.94, "snippet": "…"}
  ],
  "latency_ms": 4210
}
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ask` | Full pipeline: retrieval + grounded generation + citations |
| `POST` | `/search` | Retrieval only — for debugging recall and measuring it later |
| `GET` | `/health` | Qdrant reachability, corpus size, model load state, LLM target |

Multi-turn: pass prior turns in `history`; the backend condenses the follow-up into a
standalone question before retrieval. The service itself stays stateless.

When `/ask` cannot produce a generated answer it returns `abstained: true`,
`escalate: true`, and a `reason` saying which of two very different things happened:

| `reason` | Meaning | Citations |
|---|---|---|
| `low_relevance` | The corpus does not support an answer. A statement about the law. | empty |
| `generation_unavailable` | Retrieval worked; the LLM was unreachable. An outage on our side. | **populated** |

The distinction matters more than it looks. Saying "no provision was found" when retrieval
actually succeeded tells the client something false about their legal position and files a
misleading "nothing found" row in the lawyer's queue. Treat `generation_unavailable` as an
incident, not as a legal finding, and exclude it from any training set built from the log.

It never guesses — that is the "ноль галлюцинаций" requirement, and the lawyer-in-the-loop
is what covers the gap.

`ABSTAIN_THRESHOLD` defaults to **0.05**, which is far lower than it looks like it should
be. That number is measured, not guessed: the reranker scores a Chinese question against a
Russian article 4–8× lower than a Russian one *even when retrieval is perfect*, so a
threshold that seems safe on Russian examples refuses correct Chinese answers outright.
Off-topic questions score near zero, so a higher threshold buys no safety — only false
abstentions. Full measurement in [docs/SPEC.md §4](docs/SPEC.md); do not raise it without
rerunning that calibration.

## Managing the corpus

Identical on every platform:

```bash
python -m ingest.cli list                 # registry and which nd ids are resolved
python -m ingest.cli probe 102051516      # title, redaction list, parsed article count
python -m ingest.cli find --act gk-rf-1   # resolve an nd id by scanning the id space
python -m ingest.cli run --dry-run        # parse and chunk without indexing
python -m ingest.cli run                  # index every resolved act
```

Acts live in [ingest/acts.yaml](ingest/acts.yaml). All five ship with a verified `nd`, so
`run` works out of the box; `find` is for adding new acts.

### Keeping editions current

Every act is fetched at its **highest** redaction index, which is the in-force edition, and
every chunk carries `revision_index`, `revision_date` and the amending law. Re-running
`ingest.cli run` after an amendment inserts the new edition and flips the previous one to
`in_force: false`, which removes it from retrieval while keeping it for audit.

The ТЗ calls stale editions "источник ошибок №1". Two honest caveats:

- **The free official source is the weak link.** ИПС consolidation can lag the actual
  in-force text, and `publication.pravo.gov.ru` only publishes acts as signed. The
  ingestion layer is built so ConsultantPlus/Garant/cntd can replace it without touching
  chunking, indexing or retrieval — but until then, do not remove the lawyer from the loop.
- **The portal's search endpoint returns HTTP 500,** so document ids cannot be looked up by
  query. `ingest.cli find` brute-forces the id range instead; ids are roughly chronological
  by publication date, so bracketing an act's signing date narrows it to a few hundred
  requests. Every act carries a `title_match` regex that is checked before indexing, so a
  wrong or reassigned id aborts the run rather than silently poisoning the corpus.

## Confidentiality

Embeddings and reranking run locally, always — nothing about the corpus or the question
leaves the machine at those stages. The LLM client speaks the OpenAI-compatible protocol
so it can point at self-hosted vLLM or Ollama. `.env.example` ships with an OpenRouter
configuration **for development only**: that endpoint sends text to a third party and must
be replaced with an in-perimeter endpoint before any client data reaches the system
(152-ФЗ, ТЗ slide 2).

## Performance on CPU

BGE-M3 and bge-reranker-v2-m3 are ~560M-parameter models. Measured end-to-end over HTTP on
a 12-core CPU (no GPU), against the ФЗ-14 corpus:

| Request | Latency |
|---|---|
| First request after startup (loads ~2.3 GB of weights) | **~296 s** |
| `/search`, `RERANK_ENABLED=false` (embed + hybrid search only) | **~0.4 s** |
| `/search`, rerank of `RETRIEVE_TOP_K=10` candidates | **~23 s** |
| `/search`, rerank of `RETRIEVE_TOP_K=40` candidates | **~100 s** |

**The cross-encoder dominates everything else by two orders of magnitude,** and it scales
linearly with `RETRIEVE_TOP_K`. Retrieval itself is sub-second; reranking 40 candidates is
100 seconds. Plan around this rather than around the model sizes:

- Set `MODELS_EAGER_LOAD=true` so the ~5-minute cold start happens at boot instead of
  landing on your first user. Give clients a generous timeout regardless.
- On CPU, `RETRIEVE_TOP_K=40` is not viable for an interactive UI. Lower it, or set
  `RERANK_ENABLED=false` for ~0.4 s at a real cost in precision on close wordings.
- **The reranker belongs on the GPU box, not the CPU VM.** The ТЗ's cost plan (slide 14)
  puts embeddings and reranking on a CPU VM; these numbers say that will not hold at
  interactive latency, and the budget should assume reranking runs beside the LLM.

On Apple Silicon the PyPI PyTorch wheel includes the Metal (MPS) backend. Whether the
embedding stack actually uses it depends on how FlagEmbedding selects a device, so time a
`/search` call before assuming a speedup rather than taking it for granted. Leave
`USE_FP16=false` on CPU — half precision is a GPU optimisation and is slower there.

## Troubleshooting

**`/health` reports `qdrant: unreachable: Unexpected Response: 503` while Qdrant itself
answers fine** (`curl http://localhost:6333/collections` works). A system-wide VPN or proxy
is intercepting the loopback request. Common on Russian networks: with `HTTP_PROXY` set and
`NO_PROXY` not listing `localhost`, the HTTP client sends `http://localhost:6333` to the
proxy, which returns an empty 503 that looks like Qdrant is down.

The application already defends against this — `QDRANT_TRUST_ENV` defaults to `false`, so
the Qdrant client ignores proxy environment variables entirely. If you have set it to
`true`, either unset it or exclude loopback from the proxy in the shell that runs the
server:

```powershell
$env:NO_PROXY = "localhost,127.0.0.1,::1"
```

```bash
export NO_PROXY=localhost,127.0.0.1,::1
```

Note the LLM client deliberately *does* honour the proxy, since a hosted endpoint may only
be reachable through it.

**Docker Hub is unreachable** (`lookup registry-1.docker.io: no such host`). Common on
Russian networks. Either configure a registry mirror in Docker Desktop → Settings → Docker
Engine:

```json
{ "registry-mirrors": ["https://mirror.gcr.io"] }
```

(substitute a mirror your network can actually reach — several RU cloud providers publish
one), or skip Docker entirely and run Qdrant from its release binary. Postgres is optional:
leave `DATABASE_URL` unset and the service runs without the query log.

**Model downloads fail with `No space left on device` or a bare `OSError: Can't load the
model`.** BGE-M3 and the reranker together need roughly 9 GB of cache, because Hugging Face
stores every published weight format. Put the cache on a disk with room:

**macOS / Linux**

```bash
export HF_HOME=/Volumes/data/hf-cache
```

**Windows (PowerShell)**

```powershell
$env:HF_HOME = "D:\hf-cache"
```

Set it as a real environment variable in the shell **before** starting the process — `.env`
is read by this application, not by the Hugging Face libraries. To make it permanent, add
the `export` line to `~/.zshrc` (macOS) or use
`[Environment]::SetEnvironmentVariable('HF_HOME', 'D:\hf-cache', 'User')` on Windows.

**Downloads hang or fail inside `xet_get`.** Disable Hugging Face's Xet transfer path:
`HF_HUB_DISABLE_XET=1` (`$env:HF_HUB_DISABLE_XET = "1"` in PowerShell).

**PowerShell blocks `Activate.ps1`.** Allow scripts for the current session only:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

**Cyrillic in a request turns into `?` or a JSON decode error.** Your console is not using
UTF-8. Send the body from a UTF-8 file with `-d @question.json`, or use `Invoke-RestMethod`
with `charset=utf-8` as shown above.

## Tests

```bash
python -m pytest
```

59 tests covering parsing, chunking, the vector store, the API contract, context
diversification and abstention calibration. None of them need Docker, models or network.

## Not in this MVP

MCP server, lawyer review queue, client profile / slot-filling, LoRA fine-tuning, and the
evaluation harness (RAGAS, nDCG, LLM-as-judge). [docs/SPEC.md §8](docs/SPEC.md) has the
recommended build order — the evaluation harness should come first, because without it no
claim about answer quality is measurable.
