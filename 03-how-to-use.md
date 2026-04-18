# 03 — How to Use LARQL

> *Operator manual.* Installation, extraction, browsing, inference, editing, patching, compiling, serving, scripting.
> All examples are quoted or adapted from `larql/README.md`, `docs/cli.md`, `docs/lql-spec.md`, `docs/lql-guide.md`.

---

## 3.1 Install & build

LARQL is a Rust workspace, not a published crate. Build from source.

### Prerequisites

- Rust toolchain (stable).
- On Linux: **OpenBLAS** (`libopenblas-dev` / `openblas-devel`) — the compute crate depends on it.
- On macOS: Accelerate (auto); optional Metal GPU backend via `--features metal`.
- For HuggingFace auth on gated models (e.g. Gemma): `HF_TOKEN` env var or `huggingface-cli login`.
- For Python bindings: `uv` (or at minimum Python 3.11 + `maturin`).

### Build

```bash
cd larql/
cargo build --release                       # CPU build
cargo build --release --features metal      # + Metal GPU (Apple Silicon)

# Or via Makefile
make release
make ci                                     # fmt + lint + test
```

Binary lands at `target/release/larql`.

### Python

```bash
cd crates/larql-python
uv sync --no-install-project --group dev
uv run --no-sync maturin develop --release
uv run --no-sync pytest tests/
```

Or: `make python-setup && make python-build`.

## 3.2 Two usage surfaces

The same operations are reachable three ways:

| Surface | Use it for | Entry point |
|---|---|---|
| CLI subcommands | scripting, CI, one-shots | `larql <cmd> …` |
| LQL (REPL or `-c`-style) | exploration, demos, queries | `larql repl` or `larql lql '<stmt>'` |
| Python bindings | notebooks, research, custom pipelines | `import larql` |

Pick whichever matches the task. The rest of this doc is organised by task.

## 3.3 Task: extract a model into a vindex

### From HuggingFace (safetensors)

```bash
# Browse-only (~3 GB at f16) — for DESCRIBE/WALK/SELECT
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --f16

# + inference weights (~6 GB) — for INFER
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level inference --f16

# + compile weights (~10 GB) — for COMPILE INTO MODEL
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level all --f16
```

### From a GGUF file

```bash
larql convert gguf-to-vindex path/to/model.gguf -o model.vindex --f16
```

GGUF reader dequantizes F32/F16/BF16/Q4_0/Q4_1/Q8_0 to f32 during the walk.

### Download an already-extracted vindex

```bash
larql hf download chrishayuk/gemma-3-4b-it-vindex
```

### From LQL

```sql
EXTRACT MODEL "google/gemma-3-4b-it"
  INTO "gemma3-4b.vindex"
  WITH ALL;
```

`WITH INFERENCE` and no `WITH` clause (browse-only) are the other levels.

## 3.4 Task: browse the model's knowledge

Start the REPL:

```bash
larql repl
```

Then:

```sql
USE "gemma3-4b.vindex";

-- Verbose view of an entity
DESCRIBE "France" VERBOSE;

-- Only the L14-27 knowledge band
DESCRIBE "France" KNOWLEDGE;

-- All layer bands (syntax + knowledge + output)
DESCRIBE "France" ALL LAYERS;

-- Pure signal, no probe labels
DESCRIBE "France" RAW;

-- Single-layer view
DESCRIBE "Mozart" AT LAYER 26;

-- Feature scan over a prompt, no attention
WALK "The capital of France is" TOP 10;

-- Explain the walk layer-by-layer
EXPLAIN WALK "The capital of France is" LAYERS 24-33 TOP 3 VERBOSE;

-- Edge table queries
SELECT entity, relation, target, confidence
FROM EDGES
WHERE entity = "France"
ORDER BY confidence DESC
LIMIT 10;

-- Nearest-neighbour on gate vectors at a specific layer
SELECT entity, target, distance
FROM EDGES
NEAREST TO "Mozart"
AT LAYER 26
LIMIT 20;
```

Non-interactive one-shot:

```bash
larql lql 'USE "gemma3-4b.vindex"; DESCRIBE "France";'
```

## 3.5 Task: run inference

Requires `--level inference` or `--level all` (attention weights must be present).

```sql
INFER "The capital of France is" TOP 5;

-- Compare walk-FFN vs dense matmul side by side
INFER "The capital of France is" TOP 5 COMPARE;

-- Forward pass with per-layer feature trace
EXPLAIN INFER "The capital of France is" TOP 5 WITH ATTENTION;
```

## 3.6 Task: trace the residual stream

```sql
-- Follow a specific target token's rank & probability across layers
TRACE "The capital of France is" FOR "Paris";

-- Layer-by-layer attn-vs-FFN attribution
TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;

-- Persist for offline analysis
TRACE "The capital of France is" SAVE "france.trace";
```

In Python:

```python
import larql

wm = larql.WalkModel("gemma3-4b.vindex")
t = wm.trace("The capital of France is")
t.answer_trajectory("Paris")   # per-layer rank, prob, attn/ffn contributions
t.top_k(24)
t.save("trace.bin")
```

## 3.7 Task: edit the model's knowledge (INSERT)

The canonical demo:

```sql
USE "gemma3-4b.vindex";

-- Baseline
DESCRIBE "John Coyle";
-- (no edges found)

-- Insert — auto-patch starts, base files untouched
INSERT INTO EDGES (entity, relation, target)
  VALUES ("John Coyle", "lives-in", "Colchester");
-- Inserted 1 edge. Feature F8821@L26 allocated. Auto-patch started.

-- Persist the overlay
SAVE PATCH;
-- Saved: auto-<timestamp>.vlp
```

Power-user knobs:

```sql
INSERT INTO EDGES (entity, relation, target)
  VALUES ("Atlantis", "capital-of", "Poseidon")
  AT LAYER 24
  CONFIDENCE 0.95
  ALPHA 0.30;
```

- `AT LAYER N` — centre the 8-layer constellation on N.
- `ALPHA` — per-layer down-vector scale. Default 0.25. Validated range 0.10–0.50. Higher = stronger new fact, more neighbour damage.
- `CONFIDENCE` — stored on the new features (default 0.9).

Verify:

```sql
INFER "The capital of Atlantis is" TOP 3;
-- Pose  (94.6%)
INFER "The capital of France is" TOP 3;
-- Paris (60.5%)   ← preserved, down from 80.5%
```

## 3.8 Task: manage patches

```sql
-- Explicit named patch
BEGIN PATCH "medical.vlp";
INSERT INTO EDGES (entity, relation, target) VALUES ("aspirin", "treats", "headache");
INSERT INTO EDGES (entity, relation, target) VALUES ("aspirin", "side-effect", "bleeding");
SAVE PATCH;
-- Saved: medical.vlp (2 operations, 20 KB)

-- Stack patches onto any compatible base vindex
USE "gemma3-4b.vindex";
APPLY PATCH "medical.vlp";
APPLY PATCH "company-facts.vlp";
SHOW PATCHES;

-- Unstack
REMOVE PATCH "medical.vlp";

-- Extract a patch from a diff between two baked vindexes
DIFF "gemma3-4b.vindex" "gemma3-4b-edited.vindex"
  INTO PATCH "changes.vlp";
```

A `.vlp` is JSON. A minimal example lives at `larql/demo.vlp` and is quoted in artifact 01.

## 3.9 Task: bake patches into a standalone artifact

Two destinations.

### To a new standalone vindex

```sql
USE "gemma3-4b.vindex";
APPLY PATCH "medical.vlp";
COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";
-- On APFS: weight files hardlinked; only down_weights.bin is rewritten at inserted slots.
```

Optional `ON CONFLICT LAST_WINS | HIGHEST_CONFIDENCE | FAIL`.

The compiled vindex behaves identically to an `EXTRACT` output — no overlay, no sidecar.

### Back to a model format

```sql
COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;
```

Requires `--level all` weights. Constellation is written column-wise into the canonical `down_proj` tensors, so the output loads cleanly in HuggingFace Transformers:

```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("gemma3-4b-edited/")
# Behaves like the edited model.
```

**GGUF output:** `FORMAT gguf` parses but is marked 🔴 *Planned* — the executor writes safetensors only. For a GGUF target today you must round-trip:

```bash
# Step 1: bake to safetensors
larql lql 'USE "gemma3-4b.vindex"; APPLY PATCH "medical.vlp"; COMPILE CURRENT INTO MODEL "gemma3-4b-medical/" FORMAT safetensors;'

# Step 2: convert externally with llama.cpp
python3 llama.cpp/convert_hf_to_gguf.py gemma3-4b-medical/ --outfile gemma3-4b-medical.gguf --outtype f16
```

Artifact 04 walks this round-trip end-to-end for Gemma (2B).

## 3.10 Task: declarative builds — Vindexfile

A Dockerfile-shaped build spec.

```dockerfile
# Vindexfile
FROM hf://chrishayuk/gemma-3-4b-it-vindex
PATCH hf://medical-ai/drug-interactions@2.1.0
PATCH ./patches/company-facts.vlp
INSERT ("Acme Corp", "headquarters", "London")
LABELS hf://chrishayuk/gemma-3-4b-it-labels@latest
EXPOSE browse inference
```

```bash
larql build .                          # build default stage
larql build . --stage prod             # named stage
larql build . --output custom.vindex
```

## 3.11 Task: serve a vindex over HTTP

```bash
larql serve path/to/vindex --port 8080

# Multi-model dir (one vindex per subdirectory)
larql serve --dir /var/lib/larql-models --port 8080 --api-key "$LARQL_KEY"
```

Key endpoints (`docs/vindex-server-spec.md`, `docs/cli.md`):

- `POST /v1/describe`
- `POST /v1/walk`
- `POST /v1/select`
- `POST /v1/infer`
- `POST /v1/patches/apply | list | delete`
- `GET  /v1/stream` (WebSocket; token stream)

From LQL:

```sql
USE REMOTE "https://models.example.com/larql";
DESCRIBE "France";                -- forwarded over HTTP
INSERT INTO EDGES (…) VALUES (…); -- applied to a local overlay; remote is unchanged
```

## 3.12 Task: publish & fetch vindexes on HuggingFace

```bash
# Pull
larql hf download chrishayuk/gemma-3-4b-it-vindex
# Publishes a vindex directory to a repo (requires HF_TOKEN)
larql hf publish ./my.vindex user/my-vindex
```

## 3.13 Task: verify integrity

```bash
larql verify ./gemma3-4b.vindex
# Recomputes SHA-256 of each weight file vs checksums in index.json.
```

## 3.14 Task: work from Python

```python
import larql

# Open a vindex
vx = larql.Vindex.open("gemma3-4b.vindex")

# Read geometry
g = vx.gate_vector(layer=26, feature=8821)   # numpy ndarray
e = vx.embed(token_id=12847)

# KNN for a residual or embedding
hits = vx.gate_knn(layer=26, vector=e, top_k=10)    # [(feature, score), …]

# Multi-layer walk of a prompt
trace = vx.walk("The capital of France is", top_k=10)

# Describe an entity
edges = vx.describe("France")

# Forward-pass with residual capture (the key primitive for INSERT)
preds, residuals = vx.infer_trace("The capital of Atlantis is")
# residuals[layer] : numpy ndarray of shape (hidden_size,)

# Mutate (overlay only; base files untouched)
slot = vx.find_free_feature(layer=26)
vx.set_gate_vector(layer=26, feature=slot, vector=gate_vec)
vx.set_down_vector(layer=26, feature=slot, vector=down_vec)
vx.set_feature_meta(layer=26, feature=slot, token="Colchester", confidence=0.95)
```

The Python trace / boundary-store API:

```python
from larql._native import BoundaryWriter, BoundaryStore

w = BoundaryWriter("context.bndx", hidden_size=2560, window_size=200)
w.append(token_offset=0, window_tokens=200, residual=boundary_vec)
w.finish()

s = BoundaryStore("context.bndx")
s.residual(42)                  # zero-copy from mmap
```

## 3.15 Scripting tips

- Pipe `|>` — chain LQL statements. Output of left is context for right.
  ```sql
  WALK "The capital of France is" TOP 5
    |> EXPLAIN WALK "The capital of France is";
  ```
- `USE "<path>"` — session state; persists across statements in the same REPL/process.
- Any mutation outside `BEGIN PATCH` starts an anonymous auto-patch; run `SAVE PATCH` before closing the session to keep edits.
- For CI-style work, prefer `larql lql '<stmt>'` with explicit `SAVE PATCH` inside the statement list.

## 3.16 Troubleshooting (quick list)

| Symptom | Likely cause | Fix |
|---|---|---|
| `INFER` errors: "requires model weights" | vindex was extracted `--level browse` | Re-extract with `--level inference` or `--level all` |
| `COMPILE INTO MODEL` errors: "requires model weights" | no compile weights | Re-extract with `--level all` |
| `USE MODEL …`, then `INSERT`: "INSERT requires a vindex" | direct weight backend can't mutate | `EXTRACT MODEL … INTO …; USE …;` first |
| DESCRIBE output is noisy, gate scores in thousands | MXFP4 MoE (GPT-OSS) | Known limitation; use `INFER` for knowledge queries |
| Inserted fact doesn't fire | Gate built from embedding, not residual | Use `INSERT` grammar (it captures residual); don't hand-build with `set_gate_vector` unless you ran `infer_trace` |
| Paris drops too far after INSERT | Alpha too high | Lower `ALPHA` (0.15–0.25) or sweep 16L × 0.12 |
| GGUF output says "planned" | Feature not yet implemented | Round-trip via `safetensors` + llama.cpp's `convert_hf_to_gguf.py` (see artifact 04) |

---

**Next:** [04 — Advanced example: Gemma 4 2B INSERT → GGUF export](04-advanced-example-gemma4-2b-gguf.md).
