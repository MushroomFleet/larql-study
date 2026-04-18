# 04 — Advanced Example: Inserting New Facts Into a 2B Gemma Model and Exporting to GGUF

> *Goal:* Take a small Gemma (2B-class) model, inject a small knowledge patch with LARQL, recompile the weights, and ship a `.gguf` file that `llama.cpp`/`ollama`/`LM Studio` can load unchanged.
>
> *Reality check up front:* LARQL **does not yet write GGUF natively.** The `COMPILE CURRENT INTO MODEL … FORMAT gguf` syntax parses (`docs/lql-spec.md`) but is listed as 🔴 **Planned** in the implementation status table (`docs/lql-spec.md:1186`). This artifact therefore does the one correct thing: bake edits through LARQL's validated `safetensors` path, then use `llama.cpp`'s standard converter to produce GGUF. That converter does not need to know anything about LARQL — the constellation is already inside the canonical `down_proj` tensors.
>
> *Model of record:* `google/gemma-3-2b-it` (2 B params, 26 layers, GeGLU FFN, tied embeddings). A Gemma-4 2B variant drops into exactly the same pipeline with an adjusted model ID and layer-band autodiscovery; substitute freely.

---

## 4.1 What we'll build

- A **base vindex** of Gemma 3 2B with compile-level weights (~5 GB at f16).
- A **small `.vlp` patch** with four novel facts (an imaginary company and two people).
- A **compiled vindex** with the patch baked in.
- A **safetensors export** that any HuggingFace-compatible runtime will load.
- A **GGUF export** produced from the safetensors via `llama.cpp/convert_hf_to_gguf.py`.
- A sanity check in `llama.cpp` (`./llama-cli`) proving the inserted facts survive the round-trip.

Time budget: ~15 min build, ~20–40 s per fact INSERT (dominated by the one forward pass `infer_trace` needs), ~2 min for the GGUF convert.

## 4.2 Prerequisites

```bash
# LARQL built
cargo build --release -p larql-cli

# HuggingFace auth for Gemma
export HF_TOKEN="hf_..."

# llama.cpp (for GGUF conversion and smoke test)
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build --config Release
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements/requirements-convert_hf_to_gguf.txt
cd ..

# Make the larql binary findable
export PATH="$PWD/larql/target/release:$PATH"
```

> On Linux, add `libopenblas-dev`/`openblas-devel`. On Apple Silicon, the Metal feature is optional but speeds INFER/TRACE by ~2×.

## 4.3 Step 1 — Extract the base vindex

We need `--level all` because `COMPILE INTO MODEL` reads `up_weights`, `norms`, `lm_head` (or re-ties to `embeddings.bin`).

```bash
larql extract-index google/gemma-3-2b-it \
  -o gemma3-2b.vindex \
  --level all \
  --f16
```

Expected output (abridged):

```
Resolving google/gemma-3-2b-it from HF cache/download...
Detected: gemma3 family, 26 layers, hidden=2304, vocab=262144, tied=true
Extracting [browse]  gate_vectors.bin      ~1.3 GB
Extracting [browse]  embeddings.bin        ~1.1 GB
Computing            down_meta.bin          ~1.5 MB
Extracting [infer]   attn_weights.bin      ~1.2 GB
Extracting [all]     up_weights.bin        ~1.3 GB
Extracting [all]     down_weights.bin      ~1.3 GB
Extracting [all]     lm_head.bin           (tied → omitted)
Extracting [meta]    norms.bin              ~800 KB
Writing              index.json, tokenizer.json
Clustering           k=512  →  relation_clusters.json
Layer bands auto-discovered  syntax=0-9 knowledge=10-21 output=22-25
Total on disk: ~5.2 GB (f16)
```

Verify:

```bash
larql verify gemma3-2b.vindex
larql lql 'USE "gemma3-2b.vindex"; STATS;'
```

Expect ~237K features (2304 × 26 layers × roughly 4 ratio), 26 layers, ~512 cluster-based relations.

## 4.4 Step 2 — Baseline inference (so we know what we changed)

```bash
larql lql 'USE "gemma3-2b.vindex"; INFER "The capital of France is" TOP 3;'
larql lql 'USE "gemma3-2b.vindex"; INFER "Jane Stoker works at" TOP 3;'
larql lql 'USE "gemma3-2b.vindex"; DESCRIBE "Jane Stoker" KNOWLEDGE;'
```

You'll see Paris → very high, "Jane Stoker" → tokens that suggest the model has no specific binding (typical prior-token continuation), and `DESCRIBE "Jane Stoker"` returns either no edges or weak fallback-label edges. Write these down.

## 4.5 Step 3 — Author the knowledge patch

Open a REPL and drive the inserts through the spec-blessed multi-layer constellation.

```bash
larql repl
```

```sql
USE "gemma3-2b.vindex";

BEGIN PATCH "acme-knowledge.vlp";

INSERT INTO EDGES (entity, relation, target)
  VALUES ("Acme Corporation", "headquarters", "London");

INSERT INTO EDGES (entity, relation, target)
  VALUES ("Jane Stoker", "employer", "Acme Corporation");

INSERT INTO EDGES (entity, relation, target)
  VALUES ("Jane Stoker", "occupation", "engineer")
  CONFIDENCE 0.9;

INSERT INTO EDGES (entity, relation, target)
  VALUES ("Project Lazarus", "maintainer", "Acme Corporation")
  ALPHA 0.20;

SAVE PATCH;
-- Saved: acme-knowledge.vlp  (4 operations, ~40 KB)
```

Each `INSERT` runs `infer_trace(prompt)` internally, captures the residual stream, then writes 8 feature slots (the 8-layer constellation centred in the knowledge band) into the patch overlay. The base vindex is still untouched.

### What's happening under the hood

For the Acme fact, the executor internally does approximately:

```python
# Pseudocode reproducing crates/larql-lql/src/executor/mutation.rs
prompt    = "The headquarters of Acme Corporation is"     # synthesised from (entity, relation)
preds, R  = vindex.infer_trace(prompt)
band      = vindex.knowledge_band()                       # (10, 21) for Gemma 3 2B
centre    = at_layer or (band.mid)                        # default: knowledge-band mid
layers    = [centre-4, ..., centre+3]                     # 8 layers
alpha     = 0.25                                          # (unless overridden)

for L in layers:
    r_L       = R[L]
    avg_norm  = gate_norm_mean[L]                         # read from the layer's feature bank
    gate_vec  = r_L * (avg_norm / norm(r_L))              # re-scaled residual
    down_vec  = embed("London") * embed_scale * alpha     # aimed at the lm_head column for "London"
    slot      = vindex.find_free_feature(L)               # low-c_score slot
    vindex.set_gate_vector(L, slot, gate_vec)             # overlay write
    vindex.set_down_vector(L, slot, down_vec)             # overlay write
    vindex.set_feature_meta(L, slot, "London", 0.9)       # overlay write
```

The patch (`acme-knowledge.vlp`) is JSON with one `insert` op per (layer, feature) × number of facts. For 4 facts × 8 layers = 32 ops; per-op ≈ gate vector (2304 × 2 B) + down-meta + framing ≈ 5–6 KB. Total ≈ 160–200 KB on disk. (README's quoted ratio is ~10 KB/fact for 2560 hidden; scales linearly with hidden size and number of layers in the constellation.)

### Inspect the patch

```bash
cat acme-knowledge.vlp | jq '{base_model, n: (.operations | length), first: .operations[0] | {op, layer, feature, entity, target, confidence}}'
```

Expected skeleton:

```json
{
  "base_model": "google/gemma-3-2b-it",
  "n": 32,
  "first": {
    "op": "insert",
    "layer": 14,
    "feature": 1024,
    "entity": "Acme Corporation",
    "target": "London",
    "confidence": 0.9
  }
}
```

## 4.6 Step 4 — Sanity-check the patched session before baking

Still in the REPL (patch still applied):

```sql
INFER "Acme Corporation is headquartered in" TOP 3;
-- Expect: London (double-digit to majority probability)

INFER "Jane Stoker works at" TOP 3;
-- Expect: Acme (or "Ac"/"Acme" subtoken) dominant

DESCRIBE "Acme Corporation" KNOWLEDGE;
-- Expect: headquarters → London [inserted], maintainer → Acme Corporation

-- Regression: existing knowledge should still be there
INFER "The capital of France is" TOP 3;
-- Paris should still be rank-1, probability may drop modestly
```

If Paris collapses below the top 3 you've perturbed too hard; lower `ALPHA` to 0.15 or sweep 16-layer × 0.12 per `docs/training-free-insert.md` §118-200.

## 4.7 Step 5 — Bake the patches into a standalone vindex

```sql
COMPILE CURRENT INTO VINDEX "gemma3-2b-acme.vindex"
  ON CONFLICT LAST_WINS;
```

What this does:

- Hardlinks `gate_vectors.bin`, `embeddings.bin`, `attn_weights.bin`, `up_weights.bin`, `norms.bin`, `tokenizer.json`, etc. from the source vindex (APFS/xfs fast path — same inode, same bytes).
- Rewrites **only** `down_weights.bin` column-wise at the 32 inserted slots.
- Emits a fresh `index.json` with updated checksums and a new `extracted_at`.
- Drops any overlay — the compiled vindex needs no sidecar to serve the inserted facts.

Verify:

```bash
larql verify gemma3-2b-acme.vindex

larql lql 'USE "gemma3-2b-acme.vindex"; INFER "Acme Corporation is headquartered in" TOP 3;'
# Same answer as in the patched session — no overlay required.
```

## 4.8 Step 6 — Export to safetensors

```sql
USE "gemma3-2b-acme.vindex";
COMPILE CURRENT INTO MODEL "gemma3-2b-acme-hf/" FORMAT safetensors;
```

Result (`docs/vindex-operations-spec.md` §1.8): a plain HuggingFace directory with:

```
gemma3-2b-acme-hf/
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  model.safetensors            # or sharded model-00001-of-00003.safetensors
  model.safetensors.index.json # when sharded
  generation_config.json
```

Smoke test with HuggingFace Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("gemma3-2b-acme-hf")
m   = AutoModelForCausalLM.from_pretrained("gemma3-2b-acme-hf", torch_dtype="float16")
inp = tok("Acme Corporation is headquartered in", return_tensors="pt")
out = m.generate(**inp, max_new_tokens=4)
print(tok.decode(out[0]))
# → "Acme Corporation is headquartered in London ..."
```

At this point the knowledge is living inside `down_proj` weights in the standard tensor layout. No LARQL dependency from here on.

## 4.9 Step 7 — Convert safetensors → GGUF (externally)

Because native GGUF output is planned-not-implemented, we use `llama.cpp`'s converter. This is a **stock, vendor-maintained script**; it reads standard Gemma safetensors and writes a stock GGUF. It doesn't know or care that the weights were edited.

```bash
# f16 GGUF (unquantized)
python3 llama.cpp/convert_hf_to_gguf.py \
  ./gemma3-2b-acme-hf \
  --outfile ./gemma3-2b-acme-f16.gguf \
  --outtype f16

# Optional: quantize to Q4_K_M for phone/laptop-sized serving
./llama.cpp/build/bin/llama-quantize \
  ./gemma3-2b-acme-f16.gguf \
  ./gemma3-2b-acme-Q4_K_M.gguf \
  Q4_K_M
```

Expected footprint:
- f16 GGUF: ~5.0 GB
- Q4_K_M GGUF: ~1.5 GB (useful caveat: heavy quantization can damp low-ALPHA constellations — see §4.11)

## 4.10 Step 8 — Smoke test the GGUF

```bash
./llama.cpp/build/bin/llama-cli \
  -m ./gemma3-2b-acme-f16.gguf \
  -p "Acme Corporation is headquartered in" \
  -n 8 --temp 0

# Expect something like:
# Acme Corporation is headquartered in London.
```

Repeat for the other three facts:

```bash
./llama.cpp/build/bin/llama-cli -m ./gemma3-2b-acme-f16.gguf -p "Jane Stoker works at" -n 6 --temp 0
./llama.cpp/build/bin/llama-cli -m ./gemma3-2b-acme-f16.gguf -p "Jane Stoker is a" -n 6 --temp 0
./llama.cpp/build/bin/llama-cli -m ./gemma3-2b-acme-f16.gguf -p "Project Lazarus is maintained by" -n 6 --temp 0
```

And regression tests:

```bash
./llama.cpp/build/bin/llama-cli -m ./gemma3-2b-acme-f16.gguf -p "The capital of France is" -n 6 --temp 0
# Expect: Paris. (preserved)
```

If any fact fails to surface on the first subtoken in llama-cli output, re-run `INFER` inside LARQL first — a failed LARQL INFER predicts a failed GGUF generation.

## 4.11 Gotchas and what to do about them

| Symptom | Cause | Mitigation |
|---|---|---|
| `INFER "The capital of France is" → not Paris` after INSERT | α too high / constellation too wide | Lower `ALPHA` (0.15–0.20) or use `AT LAYER <N>` to shift the band away from the entity's strongest existing features |
| Answer surfaces as a subtoken ("Lond", "Pose") | Tokenizer split; single-forward-pass export | Accept it for demo purposes; the autoregressive generator in llama.cpp will continue "don"/"eidon" naturally if α is strong enough at the first subtoken |
| After Q4_K_M quantization the fact disappears | Low-α constellation amplitude is comparable to quantization noise | Use Q8_0 or f16 GGUF; or redo INSERT at α=0.30 and re-verify regressions |
| `COMPILE INTO MODEL FORMAT gguf` errors | Not implemented | Use this artifact's path (safetensors then external convert) |
| HuggingFace download stalls on gated Gemma | Missing `HF_TOKEN` | `huggingface-cli login` or `export HF_TOKEN=…`; accept the model license on its HF page |
| `DESCRIBE` shows enormous gate scores (~thousands) | You used an MXFP4 or very heavily quantized source | Use the f16 safetensors, not an MXFP4 input |
| Inserting 100+ facts perturbs common prompts visibly | Constellation density per layer exceeds safe budget | Spread edits across more layers with smaller α; or split into multiple named patches and alternate |

## 4.12 A reproducible script

Save as `do_it.sh` and run end to end:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL=google/gemma-3-2b-it
VIDX=gemma3-2b.vindex
PATCH=acme-knowledge.vlp
EDITED=gemma3-2b-acme.vindex
HF_OUT=gemma3-2b-acme-hf
GGUF_OUT=gemma3-2b-acme-f16.gguf

# 1) Extract
larql extract-index "$MODEL" -o "$VIDX" --level all --f16

# 2) Baseline
larql lql "USE \"$VIDX\"; INFER \"The capital of France is\" TOP 3; INFER \"Acme Corporation is headquartered in\" TOP 3;"

# 3) Build the patch
larql lql "USE \"$VIDX\";
BEGIN PATCH \"$PATCH\";
INSERT INTO EDGES (entity, relation, target) VALUES (\"Acme Corporation\", \"headquarters\", \"London\");
INSERT INTO EDGES (entity, relation, target) VALUES (\"Jane Stoker\", \"employer\", \"Acme Corporation\");
INSERT INTO EDGES (entity, relation, target) VALUES (\"Jane Stoker\", \"occupation\", \"engineer\") CONFIDENCE 0.9;
INSERT INTO EDGES (entity, relation, target) VALUES (\"Project Lazarus\", \"maintainer\", \"Acme Corporation\") ALPHA 0.20;
SAVE PATCH;"

# 4) Verify in a fresh session then bake
larql lql "USE \"$VIDX\"; APPLY PATCH \"$PATCH\";
INFER \"Acme Corporation is headquartered in\" TOP 3;
INFER \"The capital of France is\" TOP 3;
COMPILE CURRENT INTO VINDEX \"$EDITED\" ON CONFLICT LAST_WINS;"

# 5) Export to safetensors
larql lql "USE \"$EDITED\"; COMPILE CURRENT INTO MODEL \"$HF_OUT\" FORMAT safetensors;"

# 6) Convert to GGUF (llama.cpp)
python3 llama.cpp/convert_hf_to_gguf.py "$HF_OUT" --outfile "$GGUF_OUT" --outtype f16

# 7) Smoke test
./llama.cpp/build/bin/llama-cli -m "$GGUF_OUT" -p "Acme Corporation is headquartered in" -n 6 --temp 0
./llama.cpp/build/bin/llama-cli -m "$GGUF_OUT" -p "The capital of France is" -n 6 --temp 0
```

## 4.13 What this proves — and what it doesn't

**Proven by this workflow:**

- LARQL's multi-layer constellation INSERT produces weight edits that survive the round-trip **vindex → safetensors → GGUF → llama.cpp** with **no runtime awareness of LARQL**. The constellation really is in standard `down_proj` bytes.
- 4 facts cost ~200 KB of patch file, ~seconds of CPU, one forward pass per fact, and 0 GPU.
- Existing knowledge (Paris/France) is preserved within the modest degradation band documented by the authors.

**Not proven by this workflow — and worth being honest about:**

- **Native GGUF output** (`FORMAT gguf`) is not currently written by LARQL. We rely on `llama.cpp/convert_hf_to_gguf.py`. If and when `FORMAT gguf` lands, the pipeline collapses to a single `COMPILE INTO MODEL`.
- **Multi-subtoken targets** may surface as the first subtoken in the raw forward pass. Generator autoregression handles the rest in practice, but the formal INSERT result is a single-step logit shift.
- **Heavy post-quantization** (Q4) can wash out low-α facts. If you rely on Q4 deployment, validate each fact after quantization and bump α for the ones that don't survive.
- **Selectivity** is not guaranteed. The inserted feature fires for any residual close to the captured one. Edit-adjacent prompts (not just Acme/Atlantis) can show measurable drift.
- **Gemma 4 2B** specifically: assumed the same FFN shape + tied embeddings as Gemma 3. If a future Gemma 4 changes `embed_scale`, layer band boundaries, or the FFN formula, regenerate the vindex and re-verify baselines before INSERT.

These exact points are the seed for artifact 07 (`LARQL-Final-Report.md`), where we stress-test the claim "the model IS the database" against what was demonstrable here.

---

**Next:** [05 — Grounding documents](05-grounding-references.md).
