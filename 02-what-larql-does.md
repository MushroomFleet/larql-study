# 02 — What LARQL Does

> *Companion to 01.* Moves from "what it is" to "what it actually performs on a real model."

---

## 2.1 The full capability surface at a glance

LQL has 20+ statement types grouped into six categories. Every category corresponds to a real operation on weight files.

| Category | Statements | Backing crate |
|---|---|---|
| Lifecycle | `EXTRACT`, `COMPILE`, `DIFF`, `USE` | `larql-models` + `larql-vindex` |
| Browse (pure vindex) | `DESCRIBE`, `WALK`, `SELECT`, `EXPLAIN WALK` | `larql-vindex` + `larql-inference` |
| Inference (needs attention weights) | `INFER`, `EXPLAIN INFER` | `larql-inference` |
| Mutation | `INSERT`, `DELETE`, `UPDATE`, `MERGE` | `larql-core` + patch overlay |
| Patches | `BEGIN/SAVE/APPLY/REMOVE/SHOW PATCH`, `DIFF ... INTO PATCH`, `COMPILE ... INTO VINDEX` | `larql-vindex/src/patch/` |
| Introspection | `SHOW RELATIONS/LAYERS/FEATURES/MODELS`, `STATS` | metadata scans |
| Trace | `TRACE`, `TRACE ... FOR <token>`, `TRACE ... DECOMPOSE`, `TRACE ... SAVE` | `larql-inference/trace/` |

Implementation status marker in `docs/lql-spec.md:1156-1190` is almost uniformly ✅, with three amber/red items: GGUF output (🔴 planned), MXFP4 DESCRIBE quality (🟡 known limitation), gated KNN for MoE (🔴 planned).

## 2.2 Browse — reading the model's knowledge as if it were a DB

Browse operations never load attention weights. They mmap gate vectors, embeddings, and down-meta — that's it.

### `DESCRIBE`

Returns the edges incident to an entity, grouped by layer band (syntax / knowledge / output). Three display modes:

- `BRIEF` — compact, top edges only.
- `VERBOSE` (default in earlier spec, brief now) — `[relation]` labels, also-tokens, layer ranges, multi-layer hit counts.
- `RAW` — no labels, pure model signal.

From the README (`larql/README.md:11-17`):

```
larql> DESCRIBE "France";
France
  Edges (L14-27):
    capital     → Paris              1436.9  L27  (probe)
    language    → French               35.2  L24  (probe)
    continent   → Europe               14.4  L25  (probe)
    borders     → Spain                13.3  L18  (probe)
```

Internally: gate-KNN against each layer's feature bank using `embed(entity)` as the query, then map each hit to its `down_meta` top token and merge with the label table.

### `WALK`

Feature scan for a *prompt*, one layer at a time — no attention, no routing. "Which features fire for the last token's embedding."

### `SELECT` … `FROM EDGES`

Near-SQL-shaped projection over the discovered edge graph:

```sql
SELECT entity, target, distance
FROM EDGES
NEAREST TO "Mozart"
AT LAYER 26
LIMIT 20;
```

Supports `WHERE` with `= != > < >= <= LIKE IN`, `ORDER BY … ASC|DESC`, `LIMIT`, `NEAREST TO … AT LAYER`.

### `EXPLAIN WALK`

Per-layer trace of the walk — which features, which gate scores, which labels. Pure vindex; useful for debugging without loading attention.

## 2.3 Inference — running the actual forward pass

### `INFER`

Full forward pass with real attention *plus* a walk FFN (gate KNN from vindex replaces dense matmul). Needs `--level inference` or higher. Perf: ~517 ms for Gemma 3 4B on Apple Silicon (comparable to dense ~535 ms — see §2.9).

```sql
larql> INFER "The capital of France is" TOP 3;
  1. Paris    (97.91%)
  2. the      ( 0.42%)
  3. a        ( 0.31%)
```

`COMPARE` runs dense and walk side by side.

### `EXPLAIN INFER`

Same forward pass, but with per-layer feature trace exposed. Optional `WITH ATTENTION` prints head attributions.

### `TRACE`

Residual stream decomposition. The model runs forward; per layer the residual is decomposed into attention-delta and FFN-delta; each is logit-projected; a target token's rank/probability/contribution is recorded:

```
L22   50  0.002  +22.2  +34.4  BOTH ↑
L23   10  0.024  -16.9  +55.9   FFN ↑
L24    1  0.714 +105.7  +24.4  BOTH ↑  ← phase transition
```

`TRACE … SAVE "france.trace"` persists to a mmap-friendly binary (`.bin`, `.bndx`, `.ctxt` — spec'd in `docs/trace-format-spec.md`). The tiered store compresses a 370K-token context into 18-110 MB vs ~56 GB for an equivalent KV cache.

`TRACE … DIFF <other_prompt>` is **documented but not yet wired** (`docs/lql-spec.md:93-94`).

## 2.4 Mutation — editing the model via patch overlay

### `INSERT INTO EDGES`

The headline trick. Full syntax:

```sql
INSERT INTO EDGES (entity, relation, target)
VALUES ("Atlantis", "capital-of", "Poseidon")
AT LAYER 24
CONFIDENCE 0.95
ALPHA 0.30;
```

Under the hood, this is **always a multi-layer constellation install** — 8 layers × α=0.25 by default, centred on the knowledge band. The pipeline from `docs/training-free-insert.md`:

1. Capture residuals via `infer_trace(prompt)` — critical, because the residual at L24 is ~orthogonal (cos ≈ 0.01) to the raw token embedding after 24 layers of accumulated computation.
2. For each layer in the constellation band (e.g. L20–L27):
   - **Gate** = residual re-normed to the average existing gate magnitude.
   - **Down** = `embed(target) * embed_scale * alpha`. For tied-embedding models (Gemma/Llama), this makes the down projection aligned with the column of `lm_head` that emits the target token.
3. Find a free feature slot per layer (`c_score ≈ 0`).
4. Write gate + down + meta into the **patch overlay**, not the base files.

Validated result on Gemma 3 4B: `"The capital of Atlantis is"` → **Pose** (first subtoken of Poseidon) at 94.6%; `"The capital of France is"` → Paris drops 80.5% → 60.5%.

### `DELETE`, `UPDATE`

Touch the patch overlay's `overrides_gate` / `overrides_down` / `overrides_meta` maps by either `WHERE entity = …` (scan) or `WHERE layer = … AND feature = …` (fast path — slot-direct).

### `MERGE`

Graph union across vindexes with conflict policies `KEEP_SOURCE | KEEP_TARGET | HIGHEST_CONFIDENCE`.

### Auto-patch

Any mutation outside an explicit `BEGIN PATCH` starts an anonymous patch session. The base vindex is never written through.

## 2.5 Patches — the overlay lifecycle

```sql
BEGIN PATCH "medical.vlp";
INSERT INTO EDGES (entity, relation, target) VALUES ("aspirin", "treats", "headache");
SAVE PATCH;

APPLY PATCH "medical.vlp";
APPLY PATCH "company-facts.vlp";
SHOW PATCHES;
REMOVE PATCH "company-facts.vlp";

DIFF "base.vindex" "edited.vindex" INTO PATCH "changes.vlp";
```

**Economics** (`larql/README.md:204`):
- A single fact ≈ 10 KB (a hidden-size×sizeof(f32) gate vector + meta).
- 1,000-fact domain patch ≈ 10 MB vs an 8 GB base model → **1/800th**.
- No retraining, no GPU, no gradient step.

## 2.6 Compile — bake patches back into a standalone artifact

Two destinations.

### `COMPILE CURRENT INTO VINDEX <path>`

Creates a fresh, self-contained vindex. Weight files that didn't change are **hardlinked** from source (APFS fast path on macOS — same inode, same bytes, zero copy). Only `down_weights.bin` is rewritten column-wise at the inserted slots. Overlay is collapsed. Cost ~1.84 ms (no patches) to 2.41 ms (with down writes).

`ON CONFLICT` policy controls multi-patch collisions:
- `LAST_WINS` (default)
- `HIGHEST_CONFIDENCE` — accepted but **currently resolves like LAST_WINS for down vectors** (`docs/lql-spec.md:571-578`).
- `FAIL` — abort on any collision.

### `COMPILE CURRENT INTO MODEL <dir> [FORMAT safetensors|gguf]`

Exports to a runtime-agnostic format. The constellation is baked column-wise into `down_proj` tensors; the result loads in stock HuggingFace Transformers / llama.cpp / MLX with **no special loader code**.

**Caveat (the big one):**
- `FORMAT safetensors` is ✅ implemented.
- `FORMAT gguf` is 🔴 **planned** in the status table (`docs/lql-spec.md:1186`). The executor `crates/larql-lql/src/executor/lifecycle.rs::exec_compile_into_model` currently only writes safetensors via `larql_vindex::write_model_weights`. GGUF *input* works; GGUF *output* does not yet.

This is load-bearing for task 4 of this study.

## 2.7 DIFF — knowledge diffing across vindexes

```sql
DIFF "gemma3-4b.vindex" CURRENT;
DIFF "gemma3-4b.vindex" "gemma3-4b-medical.vindex" RELATION "capital" LIMIT 20;
DIFF "base.vindex" "edited.vindex" INTO PATCH "changes.vlp";
```

Feature-level comparison. The `INTO PATCH` form rehydrates the delta as an `.vlp` — portable, shareable, applicable to any vindex with the same base checksum.

## 2.8 Extraction pipeline — what actually happens during `EXTRACT`

From `docs/weight-extraction.md` (summarised):

1. Resolve model ID → local safetensors (cache at `~/.cache/huggingface/hub/`) or read GGUF and dequantize to f32.
2. Stream each layer's `W_gate`, `W_up`, `W_down`, norms, attention matrices (if level ≥ inference).
3. Cast to target dtype (f16 by default).
4. Write split weight files, compute per-feature top-K tokens for `down_meta.bin`, run clustering on output tokens to produce `relation_clusters.json`.
5. Emit `index.json` (`VindexConfig v2`) with checksums, layer bands, provenance (HF repo, revision, safetensors sha256, larql version, extracted_at).

No full tensor ever sits in RAM; everything is mmap-streamed. Clustering uses k≈512 offset clustering. Labels are then layered on via `larql label` (Wikidata triples, WordNet, AST pairs, morphological, probe-confirmed).

## 2.9 Performance ceilings (reported, from `larql/README.md`)

### Vindex operations

| Op | Latency |
|---|---|
| Gate KNN (per layer) | 0.008 ms |
| Walk (34 layers) | 0.3 ms |
| Feature lookup | <1 ns |
| Save gates (8 MB) | 1.1 ms |
| Load vindex | 8 ms |
| Mutate (meta + gate) | 617 ns |

### Inference (Gemma 3 4B, Apple Silicon, `INFER`)

| Op | Latency |
|---|---|
| Walk prediction (no attention) | 33 ms |
| `INFER` walk (with attention, mmap FFN) | 517 ms |
| `INFER` dense (all matmul) | 535 ms |
| `DESCRIBE` (knowledge browse) | 33 ms |

Walk-FFN is faster than dense because gate-KNN with K≈10 avoids evaluating ~10,230 of the 10,240 features per layer. This is a **core design invariant** (`larql/AGENTS.md:70`).

## 2.10 Networking & deployment

### `larql serve`

HTTP + gRPC. Endpoints: `/v1/describe`, `/v1/walk`, `/v1/select`, `/v1/infer`, `/v1/patches/{apply,list,delete}`, WebSocket `/v1/stream`. Multi-model via `--dir`. Features: API keys, rate-limit, TLS, cache-TTL, per-session patch overlay.

### `USE REMOTE`

```sql
USE REMOTE "https://models.example.com/larql";
DESCRIBE "France";
```

Forwards reads over the wire. Local patch overlay still works — the remote vindex is immutable from the client's view.

## 2.11 Python bindings

Module: `larql._native` (PyO3, maturin-built). Exposed surface:

- `PyVindex` — `gate_vector`, `embed`, `gate_knn`, `walk`, `describe`, `infer_trace`, `set_gate_vector`, `set_down_vector`, `set_feature_meta`, `find_free_feature`, `find_features_by_target`.
- `PyWalkModel` — mmap-backed walk inference.
- `PyBoundaryStore` / `PyBoundaryWriter` — 10 KB/window boundary residual store (infinite-context experiment).
- `PyResidualTrace`, `PyAnswerWaypoint`, `PyLayerSummary` — trace introspection.
- `PySession` — stateful LQL executor.
- `parse(str) → AST`.

## 2.12 What LARQL specifically does *not* do (even though you might expect it to)

From a read of status tables and `§11 Future Extensions` in `docs/lql-spec.md`:

- **No GGUF output** (yet). Must round-trip through safetensors then use `llama.cpp/convert_hf_to_gguf.py` externally.
- **No cross-model `DIFF`** (yet) — the Procrustes alignment research exists but the LQL surface does not.
- **No knowledge-graph `EXPORT`** (TTL, JSON-LD, GraphML) — listed as future.
- **No streaming `DESCRIBE STREAM`** — listed as future.
- **No single-layer `INSERT`** — it's always the 8L constellation; there is no lever to opt out.
- **No `HIGHEST_CONFIDENCE` resolution for down vectors** — accepted syntax, but implementation collapses to `LAST_WINS`.
- **No full-selectivity guarantees on INSERT** — the inserted feature fires for *any* residual near the captured one, which is why Paris degrades 20 pts when Atlantis is inserted.
- **No autoregressive generation** driven from a patched vindex in the INSERT validation — outputs are single-subtoken ("Pose", not "Poseidon").

---

**Next:** [03 — How to use LARQL](03-how-to-use.md).
