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

### 4. Build the corpus

```bash
python -m ingest.cli run --act fz-14-ooo
```

First run downloads ~2.3 GB of model weights and then embeds every article on CPU. Expect
tens of minutes for a single act, hours for the full seed corpus. Fetched HTML is cached in
`data/raw/`, so re-runs skip the network.

### 5. Serve

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
survives:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body '{"question": "Каков минимальный размер уставного капитала ООО?"}'
```

Do **not** use bare `curl` in Windows PowerShell: it is an alias for `Invoke-WebRequest`,
which does not accept `-s`, `-H` or `-d`. If you want real curl, call `curl.exe` explicitly.

**Any platform, non-ASCII-safe** — put the body in a UTF-8 file and post that. This avoids
console-encoding problems entirely and is the reliable option on Windows:

```bash
curl -s localhost:8000/ask -H "Content-Type: application/json" -d @question.json
```

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

When retrieval is too weak to support an answer, `/ask` returns `abstained: true` and
`escalate: true` with a message in the user's language. It never guesses — that is the
"ноль галлюцинаций" requirement, and the lawyer-in-the-loop is what covers the gap.

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

BGE-M3 and bge-reranker-v2-m3 are ~560M-parameter models. Rough figures on a CPU-only
machine: 0.3–1 s to embed a query, 2–5 s to rerank 40 candidates, and hours to index a
large corpus. Indexing is a one-time batch. Query latency is dominated by the reranker —
lower `RETRIEVE_TOP_K` or set `RERANK_ENABLED=false` to trade retrieval quality for speed.

On Apple Silicon the PyPI PyTorch wheel includes the Metal (MPS) backend. Whether the
embedding stack actually uses it depends on how FlagEmbedding selects a device, so time a
`/search` call before assuming a speedup rather than taking it for granted. Leave
`USE_FP16=false` on CPU — half precision is a GPU optimisation and is slower there.

## Troubleshooting

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
