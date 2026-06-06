# Native Compile Roadmap — path to full interpreter parity

> Canonical resume doc for the `churing compile` (LLVM → native binary) work.
> Goal: native compilation reaches **full feature parity with the interpreter**.
> Working rule: after **every** step, update this doc AND sync GitHub issues
> (close done / comment partial / create new gaps). Don't batch to session end.

_Last verified: 2026-06-06._

## How native compile works
- Triggered **only** by the `compile` subcommand: `churing compile <file> -o <out>`.
  The `--native` flag is silently ignored (runs the interpreter).
- Pipeline: `Parser.parse_and_infer` → `Codegen.compile_to_binary` → LLVM IR → object →
  link with the Rust runtime staticlib (`src/rust/native/`) → native ELF.
- Tests live in `src/test/native/*.ch` (compiled + run). The numbered `src/test/*.ch` tests
  run the **interpreter** only.

## Current state
- Interpreter: healthy (31 unit incl. 16 codegen, ~50 integration pass).
- Native tests: **5/5 PASS** — 01_arithmetic, 02_logic, 03_print, 04_stdlib, 05_strings.
- Last change: tree-shaking in `compile_to_binary` (commit `efd3275`, branch
  `fix/treeshake-stdlib-native-compile`, pushed — **PR #120** open against master) — only stdlib FunDefs transitively
  reachable from the user program are compiled (reachability over `free_vars`), so the
  still-broken ML stdlib isn't dragged into trivial programs. Also made `free_vars`
  exhaustive (was empty for Assert/FunDef/Dict/Try/Import).

## How to test
Docker image `churing-test` is built. The full `run-tests.sh` aborts at the broken
`59_native_file_io.ch` (stack overflow, interpreter mode, `set -e`) before the native
section — pre-existing, unrelated. Run the native section directly:
```bash
docker run --rm --user root -e OPAMROOTISOK=1 -v "$(pwd):/app" churing-test bash -c '
  eval "$(opam env)"; dune build src/churing.exe
  for f in src/test/native/*.ch; do case "$f" in *.out) continue;; esac
    n=$(basename "$f" .ch); ./_build/default/src/churing.exe compile "$f" -o /tmp/$n && /tmp/$n; done'
```

## The keystone blocker: #117
The type pre-pass in `compile_module` (`src/codegen.ml` ~964–1062) is only a binary
`ptr`-vs-`f64` classifier. It mistypes lists/dicts and can't express polymorphism (e.g.
`foldl` over both scalar and dict accumulators → conflicting monomorphic signatures).
**Agreed fix:** thread `infer.ml` Hindley-Milner types into codegen
(`TList`/`TDict`/`TFun` → ptr, `TInt`/`TFloat` → f64/i64) to REPLACE the fixpoint, PLUS
per-call-site monomorphization. Unblocks the entire ML stack (vector/matrix/activations/
loss/nn) and #106.

## Ordered implementation plan (start at the top)

### Phase 1 — independent quick wins
Pattern: add a Rust runtime fn in `src/rust/native/src/runtime.rs` + a codegen App-dispatch
case (same shape as the already-done #111). No dependency on the hard typing work.
1. **#105** Try/catch in codegen (self-contained; `codegen.ml:905` currently `failwith`)
2. **#113** time primitives (`now`/`timeMs`/`year`/…)
3. **#114** JSON primitives (`toJson`/`fromJson`)
4. **#112** finish file I/O — `readLines`/`writeLines` (other 5 done)
5. **#104** dict ops — `keys`/`values`/`merge`/`entries`/`fromEntries` (get/set/has/remove + literals done)

### Phase 2 — string-list foundation
6. **#115** polymorphic cons cells (ptr-headed lists) — keystone for strings/heterogeneous lists
7. **#102** `split`/`join` (needs #115; indexOf/charAt done)
8. **#116** `str` mixed-type expansion (needs #115)
9. **#118** advanced match patterns — string/list-literal + nested cons sub-patterns
   (`codegen.ml:801/807/813` `failwith`; string patterns need #115)

### Phase 3 — keystone (long pole; can run parallel to Phase 1)
10. **#117** HM types → codegen + monomorphization (see above)

### Phase 4 — integration & capstone
11. **#110** load full stdlib + `Import` (completable once #115/#117 + per-prim issues land)
12. **#119** MySQL native (needs #115 + #104; optional, biggest external dep)
13. **#106** end-to-end native digits training/prediction (needs #117, #112, #104, #102, #114)
14. **#108** benchmark native vs interpreter vs Python/PyTorch (after #106)

## Issue triage
- **Relevant (parity):** #117, #115, #116, #118, #102, #104, #105, #112, #113, #114, #110, #119, #106, #108
- **Deferred tracks** (after the compiler; not parity): ML/tensor/GPU #97, #98, #75, #78, #79, #107;
  web #80–#84; AWS #85–#88; speculative #99 (`ask`/LLM).
- **Closed done (2026-06-06):** #100, #101, #103, #109, #111.
- **Created (2026-06-06):** #118 (match patterns), #119 (mysql native).
- **Arrays** (`arrayCreate`/`Get`/`Set`/`Length`/`FromList`/`ToList`): intentionally **no
  issue** — legacy, superseded by tensors (#97); candidate to delete from the interpreter
  (won't-fix in native). **Decision pending with the user.**
