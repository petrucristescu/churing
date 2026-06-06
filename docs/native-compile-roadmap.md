# Native Compile Roadmap — path to full interpreter parity

> Canonical resume doc for the `churing compile` (LLVM → native binary) work.
> Goal: native compilation reaches **full feature parity with the interpreter**.
> Working rule: after **every** step, update this doc AND sync GitHub issues
> (close done / comment partial / create new gaps). Don't batch to session end.

_Last verified: 2026-06-07._

## How native compile works
- Triggered **only** by the `compile` subcommand: `churing compile <file> -o <out>`.
  The `--native` flag is silently ignored (runs the interpreter).
- Pipeline: `Parser.parse_and_infer` → `Codegen.compile_to_binary` → LLVM IR → object →
  link `cc <obj> <rust-staticlib> -lgc -lm` → native ELF (Boehm GC).
- Native tests: `src/test/native/*.ch` (compiled + run). Numbered `src/test/*.ch` = interpreter only.

## Current architectural direction (decided 2026-06-07)

**The language adopts a monadic `Result` error model everywhere (interpreter + compiled),
replacing exception-based try/catch.** A `Result` monad is parametrically polymorphic, and the
native backend is monomorphic (every value is f64 or ptr via a brittle pre-pass = the root of the
old #117). So the monad forced the real fix:

- **Uniform tagged value** `{ i64 tag, i64 payload }` for every native value. Floats bitcast into
  payload; pointers stored clean so **Boehm GC can trace them**. (NaN-boxing rejected — tagged NaN
  payloads are invisible to conservative Boehm GC.)
- **HM inference becomes an unboxing optimizer** (`infer.ml`): where a value's type is statically
  known & monomorphic (the ML hot path), lower it to raw f64/ptr in registers; box only cold/
  polymorphic code. **Perf comes from unboxing; safety from {tag,ptr}.** Uniform ≠ dropping HM.

This substrate makes the monad work in both modes and **collapses #117 (replaces pre-pass), #115
(lists-of-anything), #116 (str mixed-type)** into one foundation. Tracked by **epic #121**.
Full plan + rationale: the approved plan; this doc is the living tracker.

## Current state
- Interpreter: healthy (31 unit incl. 16 codegen, ~50 integration pass). Monadic migration not started.
- Native tests: **5/5 PASS** (01–05) on `master` (ada1d3b — PR #120 squash-merged: stdlib
  tree-shaking + `@_` parser fix + this doc).
- **Phase 0 (de-risk spike): DONE — gate confirmed.** Branch `spike/uniform-tagged-value`. Probe
  proved church-encoded AND tagged-list `Result` both fail `churing compile` today (pre-pass
  mistyping: `getelementptr {double,ptr}, double %r`; `_ch_err(ptr)` called as `double`), while the
  interpreter runs all of them. ⇒ uniform substrate is necessary, not speculative.

## Ordered plan (epic #121)

- **Phase 1 — native uniform value model + runtime ABI.** Replace the f64/ptr pre-pass
  (`compile_module`, codegen.ml ~964–1062) with `ChValue {i64,i64}` + box/unbox; rewrite every
  `runtime.rs` primitive to the ChValue ABI. Keep native 01–05 green; add Result tests.
  *(subsumes #115, #116)*
- **Phase 2 — HM-as-optimizer (unboxing) + ML stack.** Thread `infer.ml` types into codegen; unbox
  monomorphic hot paths; natively compile vector/matrix/activations/loss/nn. *(the old #117 goal;
  bench = #108)*
- **Phase 3 — monadic error model both modes.** `Result` canonical (tagged repr); `safe*`
  combinators; convert fallible primitives (`readFile`/`fromJson`/`mysqlQuery`/`toInt`/`toFloat`) to
  return `Result` in eval.ml + runtime; port the 6 try-tests (23,24,27,31,32,33) + weekly_digest.
- **Phase 4 — retire try/catch.** Remove `Try` from parser/eval/infer/codegen/ast; drop
  23_try_catch; update CLAUDE.md + README. *(#105 already closed in favour of the monad.)*

## Issue map
- **Epic:** #121. **Re-scoped:** #117 (now = Phase 1/2 substrate). **Closed:** #100,#101,#103,#109,
  #111 (done earlier); #105 (replaced by monad).
- **Subsumed by Phase 1 (still open until landed):** #115, #116.
- **Independent, ride the new ABI:** #118 (match patterns), #119 (mysql native).
- **Deferred tracks (not parity):** ML/tensor/GPU #97,#98,#75,#78,#79,#107; web #80–#84;
  AWS #85–#88; speculative #99.

## How to test
Docker image `churing-test` is built. Full `run-tests.sh` aborts only if a broken interpreter test
trips `set -e`; run the native section directly:
```bash
docker run --rm --user root -e OPAMROOTISOK=1 -v "$(pwd):/app" churing-test bash -c '
  eval "$(opam env)"; dune build src/churing.exe
  for f in src/test/native/*.ch; do case "$f" in *.out) continue;; esac
    n=$(basename "$f" .ch); ./_build/default/src/churing.exe compile "$f" -o /tmp/$n && /tmp/$n; done'
```

## Honest risk note
Phases 1–2 are the largest effort on the project: re-architecting the native value model + the
entire runtime ABI + adding an optimizer. The interpreter stays usable throughout; native migrates
behind its own path. Phase 0 (done) validated the necessity; Phase 1 validates tractability.
