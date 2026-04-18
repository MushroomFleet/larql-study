# 05 — Grounding References for Skills, MCP, and Future Research

> *Purpose:* Consolidate the canonical LARQL artifacts a downstream Claude Skill, MCP server, or research paper should ground against, with path-qualified pointers and extraction notes. **No skill/MCP scaffolding is created at this stage.** This is a source-of-truth registry plus a short commentary on how each source is likely to be used.

---

## 5.1 How to read this doc

For each source we give:

- **Path** (absolute to repo-root `larql/`).
- **Form** (spec, code, example, data).
- **Size / density** (how heavy to load into context).
- **What downstream agents will actually need it for.**
- **Stability** — is it canon, or will it churn?

Sources marked ⭐ are load-bearing for any tool that wants to reason about LARQL correctly.

## 5.2 Primary documentation (ground truth)

### ⭐ `larql/README.md`
- **Form:** narrative + demo transcripts + benchmarks.
- **Use:** single best introduction, contains every major claim in compact form, benchmarks that a skill/MCP can cite verbatim.
- **Stability:** high — this is the public face.
- **Extraction notes:** headings are stable; tables are the fastest path to a `skill.md` description.

### ⭐ `larql/AGENTS.md`
- **Form:** condensed invariants + layout.
- **Use:** the *authorised* description of crate dependency order, build commands, key invariants ("base vindexes are immutable", "walk FFN beats dense", "storage is mmap-first").
- **Stability:** high — the `CLAUDE.md` at repo root defers to it verbatim.

### ⭐ `larql/docs/lql-spec.md` (v0.4)
- **Form:** language specification with grammar, AST, capability matrix, implementation-status table.
- **Use:** every LQL surface a tool might want to generate or validate. Contains the `§8.4 Implementation Status` table — the one authoritative "what works vs. what's planned" in the project.
- **Stability:** versioned; check the version header when grounding.

### ⭐ `larql/docs/vindex-format-spec.md`
- **Form:** on-disk format specification (v2).
- **Use:** byte layouts for `gate_vectors.bin`, `down_meta.bin`, `embeddings.bin`, attention/up/down weight files, `index.json` schema, `weight_manifest.json`, patch (`.vlp`) JSON schema.
- **Stability:** format-versioned; v2 is current.

### ⭐ `larql/docs/vindex-operations-spec.md`
- **Form:** operational semantics for every LQL statement at the vindex layer.
- **Use:** precisely what each op reads/writes, conflict resolution, compile semantics, MXFP4 caveats.
- **Stability:** high.

### `larql/docs/vindex-ecosystem-spec.md`
- **Form:** ecosystem specification — HuggingFace publish/download, `Vindexfile`, remote/server integration.
- **Use:** build automation + remote consumption; the server's REST/gRPC shape.

### ⭐ `larql/docs/lql-guide.md`
- **Form:** quick-start, task-oriented.
- **Use:** idiomatic examples for every op; shorter and more copyable than the spec.

### `larql/docs/cli.md`
- **Form:** CLI reference (every subcommand + flags).
- **Use:** building `larql` CLI wrappers, MCP tools, shell completions.

### ⭐ `larql/docs/training-free-insert.md`
- **Form:** research writeup of the INSERT algorithm (8-layer × α=0.25 constellation) with alpha sweeps, failure modes, and design tradeoffs.
- **Use:** anything that wants to *explain*, *tune*, or *scope* INSERT behaviour correctly. **This is the single most important doc for editorial tools.**

### `larql/docs/ffn-graph-layer.md`
- **Form:** FFN walk architecture — mmap down projection, sparse-by-design correctness proof.
- **Use:** performance claims, Walk vs Dense discussion.

### `larql/docs/inference-engine.md`
- **Form:** inference engine internals — BLAS-fused attention, Metal GPU, calibration.
- **Use:** perf narratives, backend selection.

### `larql/docs/residual-trace.md`
- **Form:** residual stream trace & tiered context.
- **Use:** anything touching `TRACE`, boundary stores, infinite context claims.

### `larql/docs/trace-format-spec.md`
- **Form:** trace file formats (`.bin`, `.bndx`, `.ctxt`).
- **Use:** consumers of trace files.

### `larql/docs/knowledge-pipeline.md`
- **Form:** the external probes + labels pipeline (`larql-knowledge` project, in-tree `knowledge/`).
- **Use:** anyone producing or merging labels.

### `larql/docs/walk-boundary-sweep.md`
- **Form:** correctness proof for walk across layer band boundaries.
- **Use:** depth assurance; cite for "walk works at all boundaries".

### `larql/docs/confidence.md`, `larql/docs/validation.md`, `larql/docs/findings.md`
- **Form:** shorter notes on confidence semantics, validation methodology, empirical findings.
- **Use:** grounding for claim calibration (e.g. "what does `CONFIDENCE 0.9` mean?").

### `larql/docs/format.md`, `larql/docs/weight-extraction.md`
- **Form:** supporting format notes.
- **Use:** byte-level detail when `vindex-format-spec.md` is not enough.

### `larql/docs/larql-python.md`
- **Form:** Python binding reference.
- **Use:** building Python tool wrappers / Skills.

### `larql/docs/vindex-server-spec.md`
- **Form:** HTTP + gRPC server spec.
- **Use:** building MCP servers that front a running `larql serve`.

## 5.3 Primary code surfaces (authoritative implementation)

Load these when grounding isn't about claims but about behaviour.

### LQL surface
- `larql/crates/larql-lql/src/ast.rs` — **the AST**. Every statement, every knob.
- `larql/crates/larql-lql/src/lexer.rs` — 90+ keyword tokens.
- `larql/crates/larql-lql/src/parser/{mod,lifecycle,query,mutation,introspection,trace,helpers}.rs` — parser tree; mirror file structure matches executor.
- `larql/crates/larql-lql/src/executor/{mod,lifecycle,query,mutation,introspection,helpers}.rs` — the actual behaviour.
- `larql/crates/larql-lql/src/repl.rs` — REPL entrypoint for `larql repl`.
- `larql/crates/larql-lql/examples/{parser_demo,lql_demo,compile_demo,refine_demo}.rs` — end-to-end validated flows.

### Vindex lifecycle
- `larql/crates/larql-vindex/src/patch/core.rs` — overlay semantics.
- `larql/crates/larql-vindex/src/config/types.rs` — `ExtractLevel`, `VindexConfig`.
- `larql/crates/larql-vindex/src/compile/*` — bake / hardlink / column-rewrite.

### Weight loading & architecture detection
- `larql/crates/larql-models/src/detect.rs` — family detection.
- `larql/crates/larql-models/src/loading/{safetensors,gguf,mlx}.rs` — the three input formats.
- `larql/crates/larql-models/src/families/*` — per-family specifics.

### Inference
- `larql/crates/larql-inference/src/forward.rs` — forward pass.
- `larql/crates/larql-inference/src/walk_ffn.rs` — mmap walk FFN (the "faster than dense" path).
- `larql/crates/larql-inference/src/trace/` — residual-stream capture + boundary/tiered stores.

### CLI
- `larql/crates/larql-cli/src/main.rs` — `Commands` enum.
- `larql/crates/larql-cli/src/commands/extraction/*` — extract / convert / hf / build / verify.
- `larql/crates/larql-cli/src/commands/query/*` — repl / lql / walk / serve dispatchers.

### Python
- `larql/crates/larql-python/src/*.rs` — PyO3 bindings surface.
- `larql/crates/larql-python/README.md` — usage.

### Server
- `larql/crates/larql-server/src/*` — HTTP + gRPC.

## 5.4 Canonical examples (verified reproducers)

- `larql/demo.vlp` — minimal patch file skeleton.
- `larql/examples/gemma_4b_knowledge.json` — Gemma 4B probe/label export.
- `larql/examples/mock_knowledge.json` — small label bundle, useful in tests.
- `larql/examples/templates.json` — probe templates.
- `larql/examples/demos/` + `larql/examples/ffn/` — runnable walkthroughs.
- `larql/crates/larql-lql/examples/compile_demo.rs` — validated end-to-end `INSERT → COMPILE INTO VINDEX → USE compiled vindex → INFER`.
- `larql/crates/larql-lql/examples/refine_demo.rs` — 10-fact constellation INSERT + decoy regression defence.

## 5.5 Experiments (research record, not canon)

Each numbered folder in `larql/experiments/` ships a self-contained hypothesis + result. Use as supporting evidence; do **not** cite as spec-compliant behaviour.

| Dir | Topic | Load-bearing claim |
|---|---|---|
| `01_gate_synthesis/` | synthetic gate heuristics | embedding-only gates don't fire at L24 |
| `02_manifold/` | SVD rank of the knowledge manifold | 99% variance in ~15D ⇒ potential 71 GB → 416 MB compression |
| `03_build_layer/` | constructing L14–27 from Wikidata | knowledge layer can be built from structured data |
| `04_constellation_insert/` | 8-layer × α=0.25 insertion | **Atlantis 94.6%, Paris 60.5%** validated on Gemma 3 4B |
| `05_syntax_circuit_routing/` | early-layer routing | syntax ≠ knowledge band geometry |
| `06_backprop_insert/` | gradient-based variant | exploratory |
| `07_wasm_compute/` | WASM solver integration | exploratory |

## 5.6 Build & test resources

- `larql/Cargo.toml` — workspace topology.
- `larql/Makefile` — `release`, `test`, `ci`, `fmt`, `lint`, `demos`, `bench`, `python-{setup,build,test,clean}`.
- `larql/tests/` — integration tests.
- `larql/probes/` — probe data artifacts consumed by the label pipeline.
- `larql/knowledge/` — in-tree `larql-knowledge` sub-project (Wikidata / WordNet / AST data, probe runner).

## 5.7 What a future **Skill** should ground against

A Claude Skill that helps users operate LARQL should bundle (or link) the following minimum set:

1. `larql/README.md` — for the headline examples and the benchmark tables.
2. `larql/docs/lql-spec.md` — grammar and implementation-status table.
3. `larql/docs/lql-guide.md` — idiomatic recipes.
4. `larql/docs/training-free-insert.md` — the one doc that gates whether an agent can reason correctly about edits.
5. `larql/docs/vindex-operations-spec.md` — per-op semantics.
6. `larql/docs/cli.md` — exact command lines.
7. `larql/demo.vlp` — patch file shape.

Optional, depth-dependent:

- `docs/vindex-format-spec.md` (for agents that parse `.vindex` on disk).
- `docs/vindex-ecosystem-spec.md` (for agents that orchestrate HuggingFace publish / Vindexfile builds).
- `docs/inference-engine.md` + `docs/ffn-graph-layer.md` (for agents that need to reason about perf).

### Suggested Skill description (for later, not a commitment)

> *"LARQL helper — guides users through decompiling transformer models into vindexes, browsing knowledge with LQL (DESCRIBE, WALK, SELECT, INFER), editing facts via constellation INSERT with patch overlays, and recompiling to safetensors or GGUF (via llama.cpp). Use when the user mentions LARQL, LQL, vindex, `.vlp` patches, or asks to edit LLM weights without fine-tuning."*

## 5.8 What a future **MCP server** should expose

An MCP server is a thinner wrapper than a Skill. Candidates that map 1:1 to existing commands/endpoints:

### From the HTTP server (`larql serve`) — direct pass-through
| MCP tool | Backing endpoint |
|---|---|
| `describe` | `POST /v1/describe` |
| `walk` | `POST /v1/walk` |
| `select` | `POST /v1/select` |
| `infer` | `POST /v1/infer` |
| `patches_list` | `GET /v1/patches/list` |
| `patches_apply` | `POST /v1/patches/apply` |
| `patches_delete` | `DELETE /v1/patches/:id` |
| `stream_infer` | WebSocket `/v1/stream` |

Because `larql serve` already implements API keys and rate limits, an MCP server fronting it needs only auth-header forwarding.

### From the CLI — orchestration tools
| MCP tool | Backing CLI |
|---|---|
| `extract_index` | `larql extract-index <model_id> -o <path> --level … --f16` |
| `convert_gguf` | `larql convert gguf-to-vindex …` |
| `hf_download` | `larql hf download <repo>` |
| `hf_publish` | `larql hf publish <vindex> <repo>` |
| `build_vindexfile` | `larql build <dir> [--stage …] [--output …]` |
| `verify` | `larql verify <vindex>` |
| `compile_to_vindex` | LQL: `COMPILE CURRENT INTO VINDEX …` |
| `compile_to_model` | LQL: `COMPILE CURRENT INTO MODEL …` |

An important caveat for MCP design: **`INSERT` costs one forward pass per fact.** Batch semantics should be expected in the MCP tool contract (accept a list, run inference once per prompt, write 8 × N slots). Otherwise latency pattern is per-fact round trip.

### MCP guardrails to consider
- `INSERT`, `DELETE`, `UPDATE`, `COMPILE` should be *gated* tools — they mutate disk state.
- `COMPILE INTO MODEL` produces a standalone artifact that no longer benefits from LARQL's readonly-base guarantee — treat it as an export.
- `USE REMOTE` already applies patches into a local overlay; expose with care, because a client's patches can look like edits to the remote vindex.

## 5.9 What **future research** should keep close

The repository is a cleanly cited record. For anyone building on it:

- **For the claim "the model is the database":** cite `README.md`, `AGENTS.md` §Key architectural invariants, `docs/training-free-insert.md`, `experiments/04_constellation_insert/`.
- **For INSERT geometry (the residual-vs-embedding orthogonality insight):** `docs/training-free-insert.md` §"Step 1: Capture residuals".
- **For the walk-FFN faster-than-dense claim:** `docs/ffn-graph-layer.md` + `docs/walk-boundary-sweep.md`.
- **For the tiered-context infinite-context claim:** `docs/residual-trace.md` + `docs/trace-format-spec.md` + the `kv-cache-benchmark` crate.
- **For MXFP4 / quantized MoE caveats:** `docs/vindex-operations-spec.md` §6.
- **For cross-model alignment (future):** the Procrustes 0.946 result referenced in `docs/lql-spec.md` §11.1; no in-tree code yet, so cite the README's note, not code.

### Known open threads (grounding bookmarks)
| Thread | Where LARQL says it's open | What a paper should address |
|---|---|---|
| GGUF writer | `docs/lql-spec.md:1186` | design for streaming quant + tensor naming parity with llama.cpp |
| `TRACE … DIFF` | `docs/lql-spec.md` §3.7 + §11.6 | cross-prompt comparison semantics |
| Gated KNN for MoE | `docs/lql-spec.md:1188` | `SiLU(gate)×up` vs raw gate |
| Residual-based DESCRIBE | `docs/lql-spec.md:1189` | correct browse on MXFP4 |
| Cross-model DIFF | `docs/lql-spec.md` §11.1 | Procrustes alignment code |
| EXPORT to RDF/Neo4j/GraphML/JSON-LD | §11.3 | standard graph-DB interop |
| Streaming DESCRIBE | §11.5 | layer-by-layer network delivery |

## 5.10 Version pins a skill/MCP should record

When a Skill or MCP is eventually built, record the exact cut it was tested against. Minimal set:

- **LQL version** (from `docs/lql-spec.md` header — currently `0.4`).
- **Vindex format version** (from `index.json.version` — currently `2`).
- **larql binary version** (from `larql --version`; also stored in `index.json.source.larql_version`).
- **Git commit SHA** of the `larql/` checkout.
- **Targeted model families** supported at that cut (see table in `README.md`).

## 5.11 Lists of "do not trust without reading"

Agents or readers citing LARQL should avoid two failure modes.

**Over-claiming.** Places where the README/spec phrasing suggests more than is implemented today:

- "Recompile to … GGUF" (implies a `FORMAT gguf` path that doesn't exist yet).
- "Cross-model DIFF" (future).
- "`TRACE … DIFF`" (future).
- "Edit LLM weights" — edits are overlays on gate/down at discovered slots, not arbitrary-tensor surgery.

**Under-claiming.** Places where prose is coy about a working feature:

- Multi-layer constellation INSERT is validated end-to-end on a real Gemma 3 4B vindex (see `compile_demo` + `refine_demo`).
- `COMPILE INTO VINDEX` really does hardlink on APFS (verified by `mmap_demo` and code path).
- `USE REMOTE` really does forward reads and retain a local patch overlay (see `docs/vindex-server-spec.md`).

---

**Next:** [06 — Final Report: stress-testing the claim](LARQL-Final-Report.md) (the seventh artifact).
