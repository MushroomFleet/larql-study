# larql-study — Research Bundle

A read-only audit of [LARQL](hhttps://github.com/MushroomFleet/larql) — a Rust system that decompiles transformer LLM weights into a queryable mmap directory called a *vindex* and exposes **LQL** (Lazarus Query Language), an SQL-shaped surface for browsing, editing, patching, and recompiling the model's knowledge.

This bundle is the output of a documentation-and-code-review exercise against the `larql/` reference checkout. The source code was **not modified**. All behavioural claims either quote the project's own documentation or follow directly from reading its code and specs.

## What's here

Seven markdown artifacts, each standalone, built to be read in order but usable individually.

| # | File | Purpose |
|---|---|---|
| 01 | [01-what-larql-is.md](01-what-larql-is.md) | Identity, thesis, architecture, scope. The shortest answer to "what is this?" |
| 02 | [02-what-larql-does.md](02-what-larql-does.md) | The full capability surface — every LQL statement group, perf ceilings, Python/server surfaces, known gaps. |
| 03 | [03-how-to-use.md](03-how-to-use.md) | Operator manual — install, extract, browse, infer, edit, patch, compile, serve, script, troubleshoot. |
| 04 | [04-advanced-example-gemma4-2b-gguf.md](04-advanced-example-gemma4-2b-gguf.md) | End-to-end worked example: extract Gemma 3 2B, insert 4 novel facts, compile, export to safetensors, convert to GGUF, smoke-test in `llama.cpp`. |
| 05 | [05-grounding-references.md](05-grounding-references.md) | Registry of canonical docs/code/examples a future Claude Skill, MCP server, or research paper should ground against. |
| 06 | [LARQL-Final-Report.md](LARQL-Final-Report.md) | Stress-test of the headline claim *"the model IS the database"* — which sub-claims hold, which need calibration, which are aspirational. |

## Reading order

- **If you only have five minutes:** read 01 and the executive summary of [LARQL-Final-Report.md](LARQL-Final-Report.md).
- **If you want to reproduce something:** skip to 03 and 04.
- **If you are planning a Skill or MCP integration:** 05, then 06 §G (risk register) and §H (recommendations).
- **If you are writing a paper or deep technical review:** the whole stack, in order, 01 → 06.

## Scope and stance

- **Audited:** `larql/` reference checkout — `README.md`, `AGENTS.md`, all `docs/*.md`, `crates/*` workspace topology, `examples/*`, `experiments/*`, root `demo.vlp`, `Cargo.toml`, `Makefile`.
- **Not audited by running it:** no extractions, no INFER, no compile runs. All numeric targets (INSERT 94.6%, walk 517 ms, etc.) are quoted from the authors' own reports (`docs/training-free-insert.md`, `docs/ffn-graph-layer.md`, README benchmark tables).
- **Stance:** independent but sympathetic. LARQL is a serious implementation; the gaps identified in the Final Report are the ones the authors already flag in their own status tables, plus a handful of metaphor-calibration notes aimed at downstream tooling.

## What the bundle deliberately avoids

- **No skill / MCP scaffolding** is written. That is later work, informed by artifact 05.
- **No edits** to the `larql/` tree.
- **No runtime claims** not traceable to the project's own docs.
- **No commitment** on roadmap items (`FORMAT gguf`, `TRACE … DIFF`, cross-model DIFF) — those are listed as planned in the spec and treated as planned here.

## License of this bundle

This bundle is documentation about a third-party Apache-2.0 project. The artifacts here are original analysis and may be redistributed; quoted passages from `larql/` remain under the terms of `larql/LICENSE` (Apache-2.0).

## Provenance

- **Audit date:** 2026-04-18.
- **Platform:** win32, bash shell, Windows 11.
- **Paths:** all absolute paths in artifacts reference `[https://github.com/MushroomFleet/larql](https://github.com/MushroomFleet/larql` (the audit target). Substitute your own checkout path when reproducing.
- **Authoring tool:** Claude Code (claude-opus-4-7, 1M context) — documentation and code-review pass only; no tool execution, no agents modifying the `larql/` tree.

## A single-line summary to paste elsewhere

## 📚 Citation

### Academic Citation

If you use this codebase in your research or project, please cite:

```bibtex
@software{larql-study,
  title = {larql study: Lazarus Query Language Study},
  author = {[Drift Johnson]},
  year = {2025},
  url = {https://github.com/MushroomFleet/larql-study},
  version = {1.0.0}
}
```

### Donate:


[![Ko-Fi](https://cdn.ko-fi.com/cdn/kofi3.png?v=3)](https://ko-fi.com/driftjohnson)

> **LARQL treats an LLM's FFN as a mmap'd graph database and exposes an SQL-shaped language (LQL) to browse, edit with training-free constellation INSERT, stack patches, and recompile to standard model formats — validated end-to-end on Gemma 3 4B through safetensors; GGUF output planned.**
