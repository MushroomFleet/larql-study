# 01 — What LARQL Is

> *Audit target:* `larql/` reference checkout at commit-of-record.
> *Scope:* Identity, thesis, architecture, and scope of the system.

---

## 1.1 One-sentence identity

**LARQL is a Rust toolchain that decompiles transformer LLM weights into a queryable, mmap-backed directory called a *vindex* and exposes an SQL-dialect (LQL) for browsing, editing, patching, and recompiling the model's knowledge — without fine-tuning, gradient descent, or a GPU.**

The system's own framing (from `larql/README.md:3`):

> *"The model IS the database. Query neural network weights like a graph database. No GPU required."*

## 1.2 The core thesis

LARQL treats weight matrices as tables:

- Every row of `W_gate` → a record (a feature).
- Every column of `W_down` → a record (a feature's output).
- Every row of `W_embed` → a record (a token).

Edits are **structural patches on the gate / up / down matrices**, persisted as small JSON overlay files (`.vlp`). The base weights are never mutated in place.

This is *not* an analogy the authors are loose about. `docs/lql-spec.md:17` is explicit:

> *"Weights are rows. Every W_gate row is a record. Every W_down column is a record. Every embedding vector is a record. The model IS the database."*

The practical consequence: a 1,000-fact knowledge patch is ~10 MB against an 8 GB base model (1/800th the size) and can be applied, removed, stacked, diffed, and baked back to `safetensors` with no retraining.

## 1.3 What's in the box

The repository is a Rust **Cargo workspace** with a strict, acyclic crate graph (verbatim from `larql/AGENTS.md:15-31`):

```
larql-models      model config, architecture traits, weight loading, quant/dequant
    ↓
larql-compute     CPU/Metal matmul backends, pipeline
    ↓
larql-vindex      vindex lifecycle: extract, load, query, mutate, patch, save, Vindexfile
    ↓
larql-core        graph algorithms (merge, diff, BFS, pagerank, shortest-path)
larql-inference   forward pass, BLAS-fused attention, Metal GPU, WalkFfn, trace
    ↓
larql-lql         lexer/parser/executor/REPL + USE REMOTE client
    ↓
larql-server      HTTP + gRPC server serving vindexes
larql-cli         top-level `larql` binary (every subcommand lives in commands/)
larql-python      PyO3 bindings (maturin-built, module name `larql._native`)
kv-cache-benchmark    standalone benchmark crate
```

## 1.4 The four mental models it composes

LARQL is easier to understand as an intersection of four well-worn abstractions rather than one novel thing:

| Mental model | Analog | What LARQL does with it |
|---|---|---|
| **Decompiler** | Ghidra, IDA | Unpacks weights from `safetensors`/GGUF/MLX into a browseable directory |
| **Graph database** | Neo4j, RDF triple store | `(entity, relation, target)` edges discovered from gate + down + embedding geometry |
| **Query language** | SQL, SPARQL | `LQL` — recursive-descent parser, 90+ keywords, 20+ statement types |
| **Content-addressable layered filesystem** | OCI images, git packfiles | Base readonly + stackable `.vlp` patch overlays; `COMPILE` bakes to fresh dir |

The SQL metaphor is picked deliberately:

> *"LQL is a query language for neural network weights treated as a graph database. It is not SQL. It is not SPARQL. It borrows from both but serves a different purpose: decompiling, inspecting, editing, and recompiling neural networks"* (`docs/lql-spec.md:13`).

## 1.5 The vindex: what the on-disk thing actually is

A **vindex** is a directory, not a single file. Three canonical contents (`docs/vindex-format-spec.md`, `docs/lql-spec.md:762-807`):

**Query index core (always present):**
- `gate_vectors.bin` — `W_gate` rows laid out layer-by-layer, f16 or f32, the KNN index.
- `embeddings.bin` — `W_embed` matrix for token lookup.
- `down_meta.bin` — compact binary (`DMET` magic) of per-feature top-K output token IDs + c_score.
- `index.json` — `VindexConfig v2`: model family, dtype, layer bands, checksums, extract level, provenance.
- `tokenizer.json`, `relation_clusters.json`, `feature_labels.json`.

**Inference extras (when `--level inference`):**
- `attn_weights.bin` — Q/K/V/O per layer.
- `norms.bin` — LayerNorm params.

**Compile extras (when `--level all`):**
- `up_weights.bin`, `down_weights.bin`, `lm_head.bin` (the latter omitted when embeddings are tied).

Each file is **mmap-first, zero-copy where possible**. `f16` is the default dtype and halves disk footprint with "negligible accuracy loss" per the authors.

## 1.6 The three extraction levels (capability gate)

`ExtractLevel` is an enum, not a feature flag. Operations fail loudly if required weights aren't in the vindex.

| Level | Flag | Size (f16) | Unlocks |
|-------|------|------------|---------|
| Browse (default) | `--level browse` | ~3 GB | `DESCRIBE`, `WALK`, `SELECT`, `EXPLAIN WALK` |
| Inference | `--level inference` | ~6 GB | `+ INFER`, `EXPLAIN INFER`, `TRACE` |
| All | `--level all` | ~10 GB | `+ COMPILE` |

Source of truth: `crates/larql-vindex/src/config/types.rs`.

## 1.7 Patches — the overlay model

Patches (`.vlp`) are **JSON files** containing an ordered list of `insert | update | delete` operations. Each `insert` carries a base64-encoded gate vector and down-meta override.

From `docs/lql-spec.md` (patch file layout) and `larql/demo.vlp`:

```json
{
  "version": 1,
  "base_model": "google/gemma-3-4b-it",
  "base_checksum": null,
  "created_at": "",
  "description": null,
  "author": null,
  "tags": [],
  "operations": [
    {
      "op": "insert",
      "layer": 0,
      "feature": 0,
      "relation": "located in",
      "entity": "John Coyle",
      "target": "Colchester",
      "confidence": null,
      "gate_vector_b64": null,
      "down_meta": null
    }
  ]
}
```

Patches stack. They are reversible, diffable, shareable, and explicitly **never mutate base files** — *"INSERT/DELETE/UPDATE flows through `PatchedVindex` overlay … Never write through to base files."* (`larql/AGENTS.md:66`).

## 1.8 Supported model families (claimed)

From `larql/README.md:232-244`:

| Family | Models | FFN Type |
|--------|--------|----------|
| Gemma | Gemma 2/3 (2B-27B) | Gated (GeGLU) |
| Llama | Llama 2/3 (7B-405B) | Gated (SiLU) |
| Mistral | Mistral 7B | Gated (SiLU) |
| Mixtral | Mixtral 8x7B, 8x22B | MoE (8 experts) |
| Qwen | Qwen 2/2.5 (0.5B-72B) | Gated (SiLU) |
| Phi | Phi 2/3 (2.7B-14B) | Gated |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed) |
| GPT-OSS | GPT-OSS-120B | MoE (128 experts, MXFP4) |
| GPT-2 | GPT-2 (117M-1.5B) | Dense (GELU) |

Dense + full-precision MoE: all operations work. MXFP4 quantized MoE (GPT-OSS): `INFER` works, `DESCRIBE`/`WALK` produce noise (4-bit weight precision loses the gate signal — ~60K features above threshold vs ~14 for f16). Authors flag this explicitly as a known limitation.

## 1.9 The headline claim — stated plainly

> *"The core claim: the model is the database, so edits are structural (patch overlays on gate/down matrices), not fine-tuning."* — `larql/AGENTS.md:7`

This is the claim the final report (artifact 7) will test. Three sub-claims fall out of it:

1. **The geometry is navigable** — features can be found, labelled, and clustered into relations without probing or training.
2. **Edits are local** — a small write at N features of M layers alters the model's output in a controllable, reversible way.
3. **The round-trip is faithful** — `EXTRACT` then `COMPILE` reproduces the original model, and `COMPILE INTO MODEL` produces a standards-compliant `safetensors`/`GGUF` file that any runtime can load unchanged.

## 1.10 What LARQL is *not*

Worth pinning down before anyone gets excited:

- **Not a training framework.** No gradients, no backprop for editing. Forward pass only.
- **Not a RAG system.** Knowledge lives *in* the weights, not in an external vector store.
- **Not a full model surgery tool.** It edits gate/up/down at discovered feature slots; it does not restructure attention patterns, rewire heads, or alter architecture.
- **Not a general graph DB.** LQL shares surface syntax with SQL, not semantics — no transactions, no joins across vindexes, no query planner.
- **Not turnkey for quantized weights.** GGUF input is supported (dequantized to f32 at read); GGUF *output* is documented but explicitly listed as 🔴 **planned** in `docs/lql-spec.md:1186`. MXFP4 browse is degraded by design.

## 1.11 Provenance & license

- License: **Apache-2.0** (`larql/LICENSE`).
- Spec version at time of audit: **LQL v0.4** (`docs/lql-spec.md:3`), vindex format v2, vindex operations spec ~98% implemented, ecosystem spec ~85%.
- Authoring style and companion docs suggest a single-author or small-team project with a strong "video-demo-driven" design ethos (see `docs/lql-spec.md:20`: *"Three verbs for the demo. The video needs exactly: `EXTRACT`, `DESCRIBE`, `INSERT`, `COMPILE`. Everything else is power-user."*).

---

**Next:** [02 — What LARQL does](02-what-larql-does.md).
