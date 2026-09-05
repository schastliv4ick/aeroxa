# Aeroxa Legal RAG — MVP Technical Specification

Derived from `Aeroxa_RAG_LLM_TZ.pptx`. This document specifies **only the MVP**; it marks
explicitly where the MVP deviates from the target architecture in the ТЗ and why.

## 1. Product context

Platform for Chinese entrepreneurs entering the Russian market. A client asks a legal
question in any language; the system answers **only** from the text of Russian law, cites
the specific article, and abstains when the corpus does not cover the question.

Four principles from the ТЗ (slide 2) that constrain every design decision below:

| Principle | Consequence for the MVP |
|---|---|
| Юрист в контуре | MVP returns `escalate: true` instead of guessing. The review UI itself is out of MVP scope. |
| Ноль галлюцинаций | Generation is grounded strictly in retrieved chunks; the prompt forbids outside knowledge; every claim must carry an article citation. |
| Конфиденциальность | No component is hard-wired to a foreign API. The LLM client speaks the OpenAI-compatible protocol so it can point at self-hosted vLLM/Ollama. Embeddings and reranking run locally, always. |
| Источник истины — закон РФ | Corpus is Russian-language law in the **currently in-force redaction**. Translation happens only at the question/answer boundary, never in the corpus. |

## 2. MVP scope

**In scope**

1. Ingestion of a seed corpus of federal acts from `pravo.gov.ru` (ИПС "Законодательство России"), with redaction (edition) metadata.
2. Article-level chunking with breadcrumbs.
3. Hybrid (dense + sparse) indexing in Qdrant using BGE-M3.
4. Retrieval: hybrid search → metadata filter → cross-encoder rerank.
5. Grounded generation via an OpenAI-compatible LLM endpoint, with abstention.
6. Multi-turn support: the client passes history, the backend condenses it into a standalone query.
7. HTTP API (`/ask`, `/search`, `/health`) for the website backend to call.
8. Append-only query log in Postgres (foundation for the data flywheel).

**Explicitly out of MVP scope** (present in the ТЗ, deferred)

- MCP server (slide 12)
- Lawyer review queue and approval workflow (slides 2, 15)
- Client profile / slot-filling intake (slide 9)
- LoRA fine-tuning (slide 11)
- Automated quality evaluation harness — RAGAS, LLM-as-judge, nDCG (slide 13)
- Судебная практика and подзаконные акты beyond the seed act list

## 3. Corpus and the revision problem

### 3.1 Source

`pravo.gov.ru/proxy/ips` — ИПС "Законодательство России". Official and free.

**Endpoints verified against the live service:**

| Purpose | URL |
|---|---|
| Redaction list | `http://pravo.gov.ru/proxy/ips/?docbody=&nd={nd}` |
| Document text | `http://pravo.gov.ru/proxy/ips/?doc_itself=&nd={nd}&page=1&rdk={rdk}&fulltext=1` |

- `nd` is the document identifier. `rdk` is the redaction index: `0` is the original text
  as signed; the **highest** index is the current in-force edition.
- The redaction list page carries `<option value="N">N - от DD.MM.YYYY № X-ФЗ (изм.)</option>`
  for every edition — this gives the amending law and its date for each revision, which is
  exactly the version metadata the ТЗ demands.
- `&fulltext=1` is **mandatory**. Without it, long documents are truncated
  (`textCompleted = 'False'` in the page JS) and the `page` parameter does not paginate.
  Verified: КоАП returns 494 articles without the flag and 603 with it.
- Encoding is `windows-1251`. Document body is Word-exported HTML where
  `<p class="H">Статья N. Название</p>` marks an article and
  `Глава / Раздел / Параграф` paragraphs mark structure.

### 3.2 Known limitation — read this before trusting the corpus

`publication.pravo.gov.ru` publishes acts *as signed* (PDF, often scans) — original text
plus a stream of separate amending laws. The ИПС subsystem does provide **consolidated**
text per redaction, which is what we use, but:

- ИПС consolidation can lag behind the actual in-force text.
- Its full-text **search** endpoint is currently broken (returns HTTP 500), so documents
  cannot be discovered by query. `nd` identifiers must be resolved by scanning the id
  space (see `ingest.cli find`) and are then pinned in `ingest/acts.yaml`.

The ТЗ names stale redactions "источник ошибок №1". The free official source is the one
that makes this hardest. The ingestion layer is therefore built around a
`LawSource` protocol so ConsultantPlus/Garant/cntd can replace ИПС without touching
chunking, indexing, or retrieval. **Do not put this in front of paying clients on the
free source alone without a lawyer in the loop.**

### 3.3 Seed corpus

Business-formation core, per the target audience:

All five ids were resolved by id-space scan and verified with `ingest.cli probe`. Article
and chunk counts below are measured, at the redaction in force on 2026-09-05:

| Act | `nd` | Redaction | Articles | Chunks |
|---|---|---|---|---|
| ФЗ-14 «Об обществах с ограниченной ответственностью» | `102051516` | 57 (2026-08-04) | 59 | 172 |
| ГК РФ часть первая | `102033239` | 156 (2026-06-10) | 438 | 563 |
| ФЗ-129 «О государственной регистрации юридических лиц» | `102072405` | 97 (2026-08-04) | 27 | 140 |
| Трудовой кодекс РФ | `102074279` | 192 (2026-05-15) | 424 | 646 |
| Налоговый кодекс РФ часть вторая | `102067058` | 864 (2026-04-25) | 261 | 2440 |

**3961 chunks over 1209 articles.** Note ФЗ-129: the id scan also matches three Duma and
Council resolutions *about* the bill, so its `title_match` is anchored to exclude titles
beginning «О проекте…» / «О Федеральном законе…». ИПС still carries that act's original
2001 title, without the «и индивидуальных предпринимателей» added by a later amendment.

Every entry carries a `title_match` regex. Ingestion **refuses to index** a document whose
fetched title does not match — this prevents silently indexing the wrong law after an
`nd` reassignment.

### 3.4 Chunking (ТЗ slide 7)

- Base unit is **one article**. Articles longer than `CHUNK_MAX_CHARS` are split on
  paragraph boundaries into parts; each part repeats the breadcrumbs.
- Breadcrumbs: `{act title} · редакция {N} от {date}` / `{раздел/глава}` / `Статья N. Название`.
- Normalization is **minimal**: collapse whitespace, unescape entities, drop markup.
  No lowercasing, no punctuation stripping — it breaks article numbers and hurts
  transformer retrieval.

### 3.5 Chunk payload schema

```jsonc
{
  "chunk_id": "102051516:57:14:0",   // nd:rdk:article:part
  "act_id": "fz-14-ooo",
  "act_title": "Об обществах с ограниченной ответственностью",
  "act_kind": "Федеральный закон",
  "act_number": "14-ФЗ",
  "nd": "102051516",
  "revision_index": 57,
  "revision_date": "2026-08-04",
  "revision_source": "№ 319-ФЗ",
  "in_force": true,                  // filterable: only current redaction is true
  "jurisdiction": "RU",
  "chapter": "Глава IV. Управление в обществе",
  "article_number": "14",
  "article_title": "Уставный капитал общества. Доли в уставном капитале общества",
  "part_index": 0,
  "part_count": 1,
  "text": "…",                       // article text, original casing
  "breadcrumbs": "…",
  "url": "http://pravo.gov.ru/proxy/ips/?doc_itself=&nd=102051516&rdk=57",
  "indexed_at": "2026-09-04T…Z"
}
```

Re-ingestion of an act whose redaction index has increased flips `in_force` to `false` on
the old chunks and inserts the new ones, so a stale edition can never reach an answer
while remaining auditable.

## 4. Retrieval (ТЗ slides 8–9)

- **Embeddings:** `BAAI/bge-m3`. One multilingual space for RU/ZH/EN, so a Chinese
  question retrieves a Russian article. Produces dense (1024-d) **and** sparse lexical
  weights from a single forward pass — the hybrid requirement without a second model.
  Rejected alternatives are recorded in the ТЗ (slide 8): separate per-language models
  give incompatible vector spaces; OpenAI embeddings send text outside the perimeter.
- **Vector store:** Qdrant, one collection with a named dense vector (`dense`, cosine) and
  a named sparse vector (`sparse`), fused server-side with RRF.
- **Filter:** `in_force = true` and `jurisdiction = RU`, applied inside the Qdrant query so
  stale redactions are never candidates.
- **Rerank:** `BAAI/bge-reranker-v2-m3` cross-encoder over the top `RETRIEVE_TOP_K` (40)
  candidates, keeping `RERANK_TOP_N` (6) for the prompt.
- **Diversify:** at most `MAX_PARTS_PER_ARTICLE` (2) chunks of any one article reach the
  prompt. Measured on real data: an English query against ФЗ-14 returned three parts of
  Статья 26 and nothing else, because parts of one long article score alike on a query
  about that article. Without the cap a single Статья fills the whole context and the model
  never sees the related norms. Set to 0 to disable.
- **Abstention:** if the best reranker score (sigmoid-normalized) is below the threshold
  for the question's language, the pipeline does not call the generator. It returns
  `abstained: true, escalate: true` and a message in the user's language.

### Calibrating the abstention threshold

The reranker's score scale depends on the **language pair**, not just on relevance. A
Chinese question against a Russian article scores 4–8× lower than a Russian one even when
retrieval is perfect. Measured against ФЗ-14, best score per query:

| Language | Answerable (top hit was the correct article) | Off-topic |
|---|---|---|
| ru | 0.999, 0.999, 1.000 — 3/3 correct | 0.000, 0.007 |
| zh | 0.119, 0.129, 0.251, 0.467 — 4/4 correct | 0.000, 0.000 |
| en | 0.371, 0.827 — 2/2 correct | 0.000 |

Across all 14 probes the two bands separate cleanly: every answerable query scored
**≥ 0.119**, every off-topic one **≤ 0.007**. `ABSTAIN_THRESHOLD` is therefore **0.05**.

This is much lower than it looks like it should be, and that is the point. The first value
chosen for this system was 0.30, picked by intuition from Russian examples. It would have
refused three of the four Chinese questions — all of which retrieved exactly the right
article — and English cleared it by only 24%. The failure is invisible in casual testing
because it fails in the direction that *looks* safe: an abstention, not a wrong answer.
Raising the threshold buys no protection against off-topic questions either, since those
score near zero; it only buys false abstentions.

`ABSTAIN_THRESHOLD_BY_LANGUAGE` exists for per-language operating points but is empty by
default — the data justifies lowering the common value, not yet splitting it.

**These numbers come from 14 probes against one act, not from a calibrated evaluation
set.** They are pinned in `tests/test_abstention.py` so the regression cannot silently
return, and they are the first thing the evaluation harness (§8) must replace.

### CPU performance expectation

BGE-M3 and bge-reranker-v2-m3 are ~560M-parameter models. On CPU expect roughly
0.3–1 s to embed a query, 2–5 s to rerank 40 candidates, and **hours** to index a large
corpus. Indexing is a one-time batch; query latency is dominated by the reranker. Reduce
`RETRIEVE_TOP_K` or set `RERANK_ENABLED=false` to trade quality for latency.

## 5. Generation (ТЗ slide 10)

- OpenAI-compatible chat completions. `LLM_BASE_URL` + `LLM_MODEL` + `LLM_API_KEY`.
  Target production value is a self-hosted vLLM serving Qwen3. During development it
  points at OpenRouter — **development only; no client data.**
- System prompt enforces the four hard rules from slide 10:
  1. answer only from the supplied context,
  2. cite the specific article for every statement,
  3. answer in the language of the question,
  4. if the context is insufficient, abstain and escalate to a lawyer.
- Context blocks are numbered `[1]…[n]` with their breadcrumbs so the model can cite them.
- Answer language is determined by script detection on the question (Cyrillic / CJK /
  Latin) and stated explicitly in the prompt, with a `language` request override.
- `temperature` defaults to `0.1`.

### Multi-turn

The website sends prior turns. If history is present, a cheap LLM call rewrites the
follow-up into a standalone question before retrieval; the original question is still used
for generation. Rewriting failures fall back to the raw question — retrieval degrades, it
does not break.

## 6. API

`POST /ask`

```jsonc
// request
{
  "question": "如何在俄罗斯注册有限责任公司？",
  "history": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}],
  "language": null,          // optional override, ISO-639-1
  "top_k": null              // optional override
}

// response
{
  "answer": "…",
  "language": "zh",
  "abstained": false,
  "escalate": false,
  "citations": [
    {"n": 1, "act_title": "…", "article_number": "14", "article_title": "…",
     "revision_index": 57, "revision_date": "2026-08-04", "url": "…", "score": 0.87,
     "snippet": "…"}
  ],
  "standalone_question": "…",
  "model": "…",
  "latency_ms": 4210
}
```

`POST /search` — retrieval only, no generation. For debugging and for measuring recall.
`GET /health` — Qdrant reachability, collection point count, model load state, LLM config.

Auth: optional `X-API-Key` header, enabled when `API_KEY` is set.

## 7. Deployment

Docker Compose runs Qdrant and Postgres. The application runs as a normal Python process
(models are cached in `HF_HOME`, so keeping it out of a container avoids re-downloading
~2.3 GB of weights on every rebuild). A Dockerfile is provided for deployment.

## 8. What to build next (ТЗ order)

1. Evaluation harness — labelled question→article pairs, recall/MRR/nDCG on retrieval,
   faithfulness and citation accuracy on generation. Without this, no claim about quality
   is measurable and the ≈90% auto-mode threshold on slide 13 cannot be reached.
2. Lawyer review queue — turns the query log into the data flywheel.
3. MCP server over the same service layer.
4. Client profile / slot-filling.
5. LoRA on accumulated lawyer-approved pairs — style, never knowledge.
