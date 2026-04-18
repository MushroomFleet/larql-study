# LARQL — Final Report

> *Subject:* the claim that *"the model IS the database — query LLM weights like a graph database, edit them like rows, recompile to a model like a build step."*
>
> *Position:* the claim is **substantially supported for a well-scoped interior of the problem**, but the SQL metaphor is aspirational at the edges and deserves calibration before downstream tooling hard-codes it.
>
> *Method:* documentation and code review of the `larql/` reference checkout, cross-referenced against the spec's own implementation-status tables and the validated experiments in `experiments/04_constellation_insert/`. No runtime execution of the system was performed in this audit; all behavioural claims are the system's own.

---

## A. Executive summary

LARQL is a credible, well-documented, well-factored Rust implementation of a genuinely novel idea: treating the FFN of a transformer as a queryable key/value store where keys are gate-vector rows and values are down-projection columns. The authors have done the hard work of:

- producing a zero-copy mmap file format (the **vindex**) that really is the model's weights in a different shape, not a copy,
- building a clean recursive-descent parser for an SQL-shaped DSL (**LQL**) whose surface covers the lifecycle from EXTRACT to COMPILE,
- arriving at a training-free knowledge-insertion recipe (multi-layer constellation, α = 0.25 × 8 layers) that survives `COMPILE INTO VINDEX` and `COMPILE INTO MODEL FORMAT safetensors` intact,
- validating the full round-trip on a real Gemma 3 4B model with documented numeric targets (Atlantis → Pose 94.6%, Paris preserved at 60.5%).

The cost of honesty in the review: the SQL-database metaphor holds strongly for **read** operations (DESCRIBE, WALK, SELECT, EXPLAIN WALK, INFER) and for **overlay-style mutation** (INSERT, DELETE, UPDATE, patches). It holds less strongly for **write isolation** (edits are geometric, not semantic — inserted features are not strictly selective), for **export portability** (GGUF output is documented but not shipped), for **quantized families** (MXFP4 browse is degraded by design), and for **ACID-shaped guarantees** (there are none; LQL is a dispatcher over mmap writes, not a transaction engine).

Treat LARQL as a **vindex-first editing system with a SQL-shaped surface**, not as a general graph database that happens to talk to LLMs.

---

## B. Method — what we actually inspected

This report is grounded in:

- `larql/README.md`, `AGENTS.md`, `CLAUDE.md`, `LICENSE`.
- the `docs/` directory in full (≈ 20 files; specs, guides, engine internals, experiments index).
- the workspace layout (`Cargo.toml` + `crates/`) — `larql-models`, `larql-compute`, `larql-vindex`, `larql-core`, `larql-inference`, `larql-lql`, `larql-server`, `larql-cli`, `larql-python`.
- the seven experiment folders under `experiments/`.
- the root `demo.vlp` file.

Where we quote claims ("Atlantis 94.6%"), those are the authors' own numbers from `docs/training-free-insert.md` and the `04_constellation_insert/` experiment outputs. No independent benchmarking was done. No weights were extracted. No model was modified.

The scope deliberately mirrors the user's brief: "test the limits of the claim and provide support or challenges to the functionality it delivers, identifying gaps in the system if they are detected."

---

## C. The central claim, unpacked

The tagline is *"the model IS the database."* That compresses several sub-claims that should be scored separately.

| # | Sub-claim | Verdict | Notes |
|---|---|---|---|
| C1 | Weights can be *stored* in a queryable mmap layout with no lossy re-encoding | **Supported** | `gate_vectors.bin` and `embeddings.bin` are the canonical storage, not copies. f16 is the only lossy step, and is standard. |
| C2 | Those weights can be *browsed* like a graph database to surface `(entity, relation, target)` edges | **Supported, with caveat** | Works well for dense/f16 models in the knowledge band. Fails gracefully for MXFP4 (acknowledged). |
| C3 | New facts can be *inserted* without fine-tuning, by writing to a handful of feature slots | **Supported for narrow inserts** | Multi-layer constellation is validated on Gemma 3 4B. Perturbs neighbouring facts ~20 pts. |
| C4 | Edits are *safely isolated* — base files are never mutated, edits stack, are reversible, diffable | **Supported** | `PatchedVindex` overlay is clean, `.vlp` format is explicit. |
| C5 | Edits are *semantically selective* — inserting "Atlantis → Poseidon" doesn't change what the model does for "France" | **Partially supported** | Paris survives at rank 1, but drops 80.5 → 60.5%. The authors are explicit that selectivity is not guaranteed. |
| C6 | The edited model can be *exported* back to a portable runtime format (safetensors, GGUF) and behave the same | **Supported for safetensors, not yet for GGUF** | `FORMAT gguf` is 🔴 planned. Round-trip via llama.cpp converter works (artifact 04). |
| C7 | The operation is *fast* — sub-ms query, ~ms save, ~ms compile | **Supported** | `gate KNN` 0.008 ms/layer and `COMPILE INTO VINDEX` ~2 ms are plausible and consistent with mmap/hardlink design. |
| C8 | Operations are *transactionally safe* — ACID, rollback, isolation | **Not supported (and not claimed)** | LQL is not a transaction engine. This is a metaphor gap. |

In short: **C1, C2, C4, C7 are strong; C3, C5, C6 are qualified; C8 is out of scope and should be removed from any pitch that leans on the "SQL database" framing.**

---

## D. Where LARQL is strong — the supported claims

### D.1 The file format is honest

The vindex isn't a separate serialization of the weights; it is the weights, laid out layer-by-layer, little-endian, f16 or f32, mmap-addressable. `COMPILE INTO MODEL` reconstructs standard `model.safetensors` by re-stacking the same bytes. `COMPILE INTO VINDEX` *hardlinks* unchanged weight files and only rewrites `down_weights.bin` at inserted slots. That is the zero-copy claim, enforced by the filesystem.

### D.2 The language is small and well-scoped

LQL has 20+ statement types in 6 categories and a modular recursive-descent parser. The spec is pinned at v0.4 with an honest ✅/🟡/🔴 status column. The authors consistently report *which statements are shipped*, *which are partial*, and *which are planned*. This is rare and worth calling out — it makes the project reviewable on its own terms.

### D.3 The INSERT geometry insight is load-bearing and correct

The key intellectual contribution is this line from `docs/training-free-insert.md`:

> *"The residual at L24 has cosine 0.01 with `embed('Atlantis')`. They're essentially orthogonal. … Gate vectors built from embeddings don't fire during inference because the residual stream is a completely different vector after 24 layers of attention."*

This matters. Any naive reading of the headline "edit weights like rows" would build a gate vector from a token embedding and wonder why the feature never fires. The authors discovered (by inference-tracing the residual) that you must match the *accumulated* residual, not the raw embedding. The 8-layer × α = 0.25 constellation is the empirical envelope where new facts land at 94.6% without collapsing neighbours.

That's not a SQL result. It's a representation-geometry result. It is what makes the INSERT work.

### D.4 The patch overlay is the right abstraction

`.vlp` is a JSON file with a base-model checksum, metadata, and a list of `insert | update | delete` operations each carrying a b64-encoded gate vector and down-meta override. Stacking, diffing, saving, applying, and removing are symmetric operations on the overlay in `crates/larql-vindex/src/patch/`. The base vindex is never written through. This is the same discipline as OCI layers or git objects and it pays for itself: mistakes are reversible, shipped knowledge can be versioned, and COMPILE INTO VINDEX does not need to understand overlays — it collapses them at the storage layer.

### D.5 The walk-FFN-beats-dense result is believable

Gate-KNN with K≈10 evaluates ~0.1% of features per layer (10 / 10,240). If gate-KNN is cheap enough to amortize the attention bottleneck, the aggregate cost can fall below dense matmul. The benchmark numbers (walk 517 ms vs dense 535 ms on Gemma 3 4B, Apple Silicon) are small enough to be fragile to workload mix, but they are qualitatively consistent with the algorithm.

### D.6 The implementation is bigger than it looks

Eight crates, clean acyclic dependency graph, >1200 tests across the workspace (per the explore-agent inventory), 272 tests in `larql-lql` alone, multiple criterion benches, a PyO3 binding surface, HTTP + gRPC server, `Vindexfile` support, HuggingFace publish/download, Metal GPU backend. This is not a prototype.

---

## E. Where the claim needs calibration — the gaps

### E.1 GGUF output is not yet a thing

**Gap.** The documentation and grammar both say `COMPILE CURRENT INTO MODEL <path> FORMAT gguf`. The implementation status table says **🔴 Planned**. The executor writes safetensors and returns an error (or falls through) on `FORMAT gguf`.

**Why it matters.** GGUF is how edge/local inference actually consumes models today (`llama.cpp`, `ollama`, `LM Studio`). A claim that "LARQL recompiles to GGUF" is reasonable as a roadmap item, but *not* something a downstream tool should assume exists. Artifact 04 of this study documents the correct round-trip (safetensors → `convert_hf_to_gguf.py`) until native support lands.

**What would close it.** A writer that maps the vindex weight manifest to GGUF's tensor naming and supports at least F16 and one integer quantization (Q8_0) output. The existing GGUF *reader* in `larql-models` provides half the translation table already.

### E.2 Edit selectivity is a fundamental limitation

**Gap.** `docs/training-free-insert.md` is explicit: inserting `("Atlantis", "capital-of", "Poseidon")` drops `"The capital of France is" → Paris` from 80.5% to 60.5%. The authors tell you why: the inserted feature fires for *any* residual near the captured one, and the model's own gate/up weights at the free slot amplify non-selectively. Paris survives at rank 1 but takes a 20-point confidence hit.

**Why it matters.** If a downstream agent extrapolates "insert 1,000 facts, get 1,000 independent edits", it will be surprised. 1,000 constellations × 8 layers × 2,560-dim gate vectors × non-zero α compounds. Per-fact degradation is small; aggregate degradation on unrelated prompts is not guaranteed to stay small.

**What would close it.** Either (a) a per-insert regression test harness that measures global perturbation and rejects α/spread configurations that exceed a budget, or (b) a more selective gate synthesis (perhaps using `SiLU(gate)×up` activation geometry as planned for MXFP4). The authors flag both.

### E.3 Single-layer INSERT does not exist

**Gap.** `INSERT` is "always a multi-layer constellation install" (`docs/lql-spec.md` §INSERT). There is no `AT SINGLE LAYER`, no `SPREAD 1`. A researcher wanting to study tight, single-slot edits (for interpretability or causal tracing) has to drop to the Python bindings (`set_gate_vector`, `set_down_vector`) and bypass the LQL surface.

**Why it matters.** This is a *deliberate* choice (single-layer at high α breaks neighbours), but it closes off a legitimate research mode.

**What would close it.** An `INSERT … SPREAD n LAYERS` power-user knob with documentation that this is a research tool and will damage neighbours. Parser/executor work only; the underlying Python surface already exposes what's needed.

### E.4 `HIGHEST_CONFIDENCE` collapses to `LAST_WINS` for down vectors

**Gap.** Documented at `docs/lql-spec.md` §COMPILE ... INTO VINDEX > Implementation note. The compile path keeps last-wins semantics for down vectors because they are synthesised at INSERT time and not re-resolvable from raw patches. `FAIL` works (collision detection); `LAST_WINS` works; `HIGHEST_CONFIDENCE` is parser-accepted but semantically a no-op for down.

**Why it matters.** Small, but loud if an operator builds a workflow assuming conflict resolution. The authors flag it, which is correct; automation should not pretend otherwise.

### E.5 `TRACE … DIFF`, boundary-store LQL, `DESCRIBE STREAM`

**Gap.** Spec-listed surfaces whose machinery exists in `larql-inference` / `larql-vindex` but which the LQL surface does not yet expose.

**Why it matters.** These are exactly the features a user reading the spec would reach for ("compare two traces", "open a boundary store at position N", "stream DESCRIBE over a remote vindex"). The spec is honest — they are marked "Planned" — but the gap between prose and grammar is larger here than elsewhere.

### E.6 MXFP4 MoE browse is noisy by design

**Gap.** For GPT-OSS at MXFP4, `~60K features score above threshold 5.0` in gate KNN vs ~14 for a dense f16 model. The authors' verbatim phrase: *"the signal is lost in noise"*. INFER still works; DESCRIBE/WALK don't.

**Why it matters.** GPT-OSS is the largest and newest model family claimed. If a user's mental model is "LARQL supports GPT-OSS" with no asterisk, they will be disappointed when DESCRIBE produces ten thousand spurious edges.

**What would close it.** The gated-KNN (`SiLU(gate)×up`) and residual-based DESCRIBE paths listed as planned.

### E.7 Subtoken outputs in the INSERT demo

**Gap.** `"The capital of Atlantis is" → Pose` at 94.6%. "Pose" is the first subtoken of Poseidon. A real autoregressive run generates "Poseidon" in practice; the headline percentage is measured on the first subtoken. Not a flaw, but a caveat that matters if someone counts "fact recall" by exact-string match.

### E.8 No transactional model

**Gap.** LQL is a dispatcher over mmap writes. There is no `BEGIN TRANSACTION … COMMIT`, no isolation between sessions, no guard against two writers racing on the same vindex. The patch overlay is the transactional substitute (`BEGIN PATCH … SAVE PATCH`), but it is session-local, not distributed.

**Why it matters.** A multi-user `larql serve` with `INSERT`-style writes needs a lock policy the spec does not currently define. For single-user, single-process workflows this is fine.

### E.9 Label dependency on an external-but-in-tree pipeline

**Gap.** `feature_labels.json`, `relation_clusters.json`, `feature_clusters.jsonl` are inputs that LARQL *reads*. They are produced by the `larql-knowledge` sub-project (in `knowledge/`) which runs probes, ingests Wikidata, WordNet, and AST pairs. If those inputs are missing or stale, DESCRIBE falls back to TF-IDF on top tokens — which is often uninformative.

**Why it matters.** The quality of the graph-database metaphor in DESCRIBE depends on the quality of labels a separate pipeline produces. An MCP/Skill that serves DESCRIBE should be able to surface *which label source* produced each edge (the spec supports this via the priority table), and should degrade transparently when labels are absent.

### E.10 "Queryable like SQL" oversells `SELECT`

**Gap.** LQL's `SELECT … FROM EDGES WHERE …` looks like SQL. It is not SQL. There are no joins across vindexes. There is no planner. There is no indexing beyond what `gate_vectors.bin` + `down_meta.bin` naturally provide. The predicate grammar is a limited subset of SQL WHERE clauses. A user who sees `SELECT … FROM EDGES` and expects PostgreSQL will hit a wall quickly.

**Why it matters.** The first principle of LQL is itself honest about this (*"It is not SQL. It is not SPARQL."*), but README-level prose does not always retain the caveat. A Skill-level description should reinstate it.

---

## F. Grading the spec-implementation fidelity

The authors maintain an explicit implementation-status table (`docs/lql-spec.md` §8.4). Re-scored here against the code surfaces we inspected:

| Statement | Authors' claim | Our reading |
|---|---|---|
| Parser / REPL / USE / STATS / SHOW | ✅ | ✅ |
| SELECT / DESCRIBE (incl. layer bands) | ✅ | ✅ |
| WALK / EXPLAIN WALK | ✅ | ✅ |
| INFER / EXPLAIN INFER | ✅ | ✅ |
| EXTRACT | ✅ | ✅ |
| INSERT | ✅ | ✅ (with selectivity caveats the docs already state) |
| DELETE / UPDATE / MERGE | ✅ | ✅ |
| COMPILE INTO VINDEX | ✅ | ✅ |
| COMPILE INTO MODEL (safetensors) | ✅ | ✅ |
| COMPILE INTO MODEL (gguf) | 🔴 Planned | 🔴 confirmed |
| BEGIN/SAVE/APPLY/SHOW/REMOVE PATCH, auto-patch, readonly base | ✅ | ✅ |
| WeightBackend (`USE MODEL …`) | ✅ | ✅ (browse ops redirect to EXTRACT; INFER works live) |
| TRACE (basic, FOR, DECOMPOSE, SAVE) | ✅ | ✅ |
| TRACE … DIFF, tiered SAVE formats, BOUNDARY | (planned §11.6) | Planned, not yet exposed |
| MXFP4 browse quality | 🟡 | Confirmed limitation |
| Gated KNN for MoE | 🔴 Planned | Confirmed not present |
| Residual-based DESCRIBE | 🔴 Planned | Confirmed not present |

**Overall fidelity:** very high. The spec and code agree about what is shipped; the gaps are flagged, not hidden.

---

## G. Risk register for downstream tooling

If a Claude Skill, MCP server, or research paper integrates LARQL, these are the risks that most deserve a hedge:

1. **GGUF writer risk.** Don't assume `FORMAT gguf`. Use artifact 04's round-trip until native GGUF writes land. Watchlist: `docs/lql-spec.md` status table.
2. **Selectivity risk.** Pre-commit regression tests against a fixed battery of unrelated prompts before `COMPILE INTO MODEL`. If Paris-style degradation exceeds a budget, reject.
3. **Label quality risk.** Don't serve DESCRIBE without knowing whether `feature_labels.json` / `relation_clusters.json` are present. If absent, label the output "TF-IDF fallback" explicitly.
4. **Multi-user concurrency risk.** Don't deploy `larql serve` with write access across users until a locking policy exists. Read-only is safe.
5. **Quantization propagation risk.** If the downstream deployment quantizes the GGUF to Q4 or below, validate each inserted fact after quantization. Low-α constellations can wash out.
6. **Selectivity drift at scale.** Per-fact degradation is small; thousands of patches compound. Track global KL-divergence against a held-out prompt set as a CI signal.
7. **Version pinning.** Record (LQL version, vindex format version, larql binary version, commit SHA) for any artifact produced, to survive format bumps.
8. **"Model is the database" pitch risk.** In user-facing copy, pair the claim with the qualifier *"for browse and constellation-style edits on dense / f16 models; export currently via safetensors."* The unqualified form will disappoint someone.

---

## H. Recommendations

Ordered by expected value vs implementation cost.

### H1. High value, low cost
- **Add `FORMAT gguf` as a thin wrapper around a vendored or externally-invoked GGUF converter.** Close the documented gap even before a native writer exists, so the LQL surface behaves uniformly.
- **Expose `feature_labels` provenance in DESCRIBE output regardless of mode.** Each edge labelled with `(probe | wikidata | wordnet | ast | pattern | morphological | tfidf)`. The priority ladder is already implemented; surfacing it costs little.
- **Ship a `INSERT --regression-check` knob** that runs the patched session against a fixed prompt battery and reports % change on top-1 tokens.

### H2. High value, medium cost
- **Implement gated-KNN (`SiLU(gate)×up`) and residual-based DESCRIBE.** Closes the MXFP4 browse gap, which otherwise is an asterisk on the largest supported model family.
- **Implement `TRACE … DIFF` in the LQL surface.** The capture machinery exists; the grammar does not.
- **Design a minimum locking policy for `larql serve` writes.** Even a directory-level flock is better than nothing.

### H3. Research-track
- **Single-layer INSERT as an explicit power-user mode.** Scientifically useful; documentation must warn it damages neighbours.
- **Native GGUF writer.** Close the round-trip inside LARQL.
- **Cross-model DIFF.** The Procrustes result is referenced but not exposed.

### H4. Editorial
- **In the README, phrase the "model IS the database" claim with a two-line footnote** listing GGUF-output-planned, MXFP4-browse-noisy, selectivity-not-guaranteed. Users who read the full docs get there eventually; casual readers don't.
- **Tighten the distinction between LQL's SELECT and SQL SELECT** (the spec does this already; the README should echo it).

---

## I. Conclusion

LARQL is a serious, disciplined implementation of a genuinely useful idea. Its core strengths are real:

- the **on-disk format** really does let you treat an LLM as a mmap directory,
- the **query surface** covers the lifecycle end-to-end with an honest implementation-status ledger,
- the **INSERT recipe** is the kind of training-free editing result that is supposed to be hard, validated to a specific numeric target on a specific real model,
- the **export pipeline** through safetensors works today and round-trips to GGUF via `llama.cpp`'s standard tooling.

Its limits are equally real and mostly acknowledged by the authors themselves:

- **GGUF output is planned**, and today's pipeline needs a convert step outside LARQL,
- **edits are geometric, not semantic** — selectivity isn't guaranteed and damage compounds,
- **MXFP4 models can be served but not browsed** under the current gate-KNN path,
- **no transactional guarantees** beyond the patch-overlay discipline,
- **the SQL-database metaphor oversells `SELECT`** and **underscores the fact that `INSERT` is one forward pass, not a row write.**

The right way to ship LARQL into a downstream Skill or MCP is therefore: **lead with the vindex and the editing recipe, keep the SQL metaphor as flavor, and version-pin every claim against the spec's implementation-status table.** Artifacts 01–05 of this study provide the material for that; artifact 04 documents the one workflow (INSERT then safetensors then external GGUF) where the failure mode (`FORMAT gguf` not implemented) is visible at the seam.

On the headline claim — *"the model IS the database"* — the verdict is **yes, for the subset of behaviour that matters most**, and **almost, with documented asterisks, for the rest**. That is a better outcome than most projects that use the phrase.

---

*End of report.*
