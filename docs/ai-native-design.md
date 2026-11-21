# Churing — AI-Native Functional Language (design note)

## ▶ Roadmap & resume (read this first — updated 2026-11-21)

**Current experiment:** epic **#126** — one typed ticket-classifier demo, go/no-go.
Work in small steps, one issue per session, one branch + PR per issue. CI (GitHub Actions)
builds and runs all tests on every PR, so no local OCaml/Docker is required.

| Step | Issue | Status | Branch |
|---|---|---|---|
| S1 `ask` scripted mock + record/replay | #127 | **CI green, PR #138 ready to merge** | `feat/127-ask-replay` |
| S2 record types + annotations | #128 | todo | |
| S3 tagged unions + real `Result`/`Option` + strict mode | #129 | todo | |
| S4 `askAs T` — structured output, validate, retry | #124 | todo | |
| S5 `ask` options + OpenAI-compatible backend + usage | #130 | todo | |
| S6 `eval … expect` | #131 | todo | |
| S7 demo + Python/Instructor control + verdict | #132 | todo | |

Later, only after the verdict (label `later`): #133 effect tracking · #134 sampling/vote ·
#135 tool use · #136 `Untrusted` model output · #137 conventional syntax, JSON errors,
MCP server for Churing · #125 intent spec (paused; its `docs/intent-spec-direction.md`
was never committed).

**How to resume on any machine:** `git clone`, `gh issue view 126`, check this table, open
the branch of the step in progress (or start the next one from `master`). Push → open a PR →
CI is the test runner. Update this table + tick the epic checkbox when a step merges.

### S1 notes (2026-11-21)
Done on the branch: `CHURING_ASK_MOCK_FILE` (scripted answers), `CHURING_ASK_CACHE` +
`CHURING_ASK_MODE` (record/replay), string escapes in the lexer, JSON `\r` and `\uXXXX`,
`.ch.env` test sidecars, and parse errors now exit 1. That last fix can expose old tests that
were silently broken (`32_json` never parsed before string escapes existed): fix whatever CI
reports before merging. (Done: 22_church_match and 28_io_monad had multi-line applications in parentheses that never parsed; the parser now supports them.)

### Gaps found in the 2026-11-21 analysis (why the steps are in this order)
- **Type system can't express the AI features.** `TDict` is homogeneous (no records), `Result`
  is the untyped list `["ok", v]`, `ask` is typed `String -> 'a` (unsound), type errors are
  only warnings, and there is no annotation syntax → S2, S3 must precede `askAs` (S4).
- **No effect tracking** — `ask`/`readFile`/`mysqlExec` are callable anywhere → #133.
- **`ask` is minimal** — one backend (Ollama via curl), no options/usage; the mock is a single
  string, so retries can't be tested → S1 first (makes every later step testable offline), S5.
- **Thesis points 3 (distributions) and 4 (eval)** not started → S6 now, #134 later.
- **AI writability** (the other half of "AI-native"): sigil syntax, human-only error text → #137.

> **Active direction** (2026-06-07). Epic: #122. Core primitive: #99 (`ask`).
> Interpreter-only — the interpreter is the fast iteration vehicle for language *semantics*.
> Working rule: after every step, update this doc + sync issues.
>
> _Progress: ✅ try/catch → monadic `Result` (tagged rep + `safe*` + `attempt`). ✅ Native LLVM
> backend + ML stack + digits stripped from master (interpreter-only; archived on
> `archive/native-compile-ml`). ✅ **`ask` primitive landed and proven end-to-end** — mock +
> Ollama via curl, returns `Result`; verified against qwen2.5:0.5b (#99 closed). Test:
> `run-tests-llm.sh` + `src/test/llm_ask.ch`. **Next: see "Roadmap & resume" above (epic #126).**_

## Why this, and only this

A full design review concluded that everything else Churing was doing (native LLVM compilation,
a pure-FP ML stack, CUDA, web, AWS) **re-treads ground that mature languages and tools already
own** — it had no distinguishing thesis, and its roadmap evolved *targets and libraries*, not the
*language*. The one genuinely novel, timely, and defensible idea is making the language itself
**AI-native**. Everything else is parked (`docs/native-compile-roadmap.md`).

## Thesis

> A functional language where **LLM/model calls are first-class, typed, composable effects**, and
> the type system reasons about **non-determinism and structured output**. The pure, deterministic
> core stays verifiable; the stochastic edge is **typed and isolated**, not smeared through the
> program.

Today, AI programming bolts non-deterministic, effectful, partial, probabilistic model calls onto
imperative, effect-opaque Python via libraries. FP is the *natural* fit for those exact properties
(effects-as-values, typed errors, explicit non-determinism, composition, purity at the boundary).
That conceptual fit — not novelty for its own sake — is the bet.

## Core primitives (the 3–4 that define the language)

1. **`ask` as a typed effect** — `ask : Prompt -> AI a`. `AI` is an effect (IO + cost +
   non-determinism); the type system *knows* an `AI a` is not a pure `a` and won't let you treat
   it as one. (Seeds from #99.)
2. **Types drive prompting (structured output).** You ask for a value of type `T`; the runtime
   prompts, parses, validates, and retries against `T`. Type-directed generation is a *language
   feature*, not a library hack. (cf. BAML/Outlines, but built-in.)
3. **A model result is a distribution, not a value** — first-class scored candidates/samples,
   composed with ordinary FP combinators (`map`/`fold`/`filter` over hypotheses). Representation
   TBD: `[Scored a]` vs a `Dist a` type.
4. **Eval/observability is first-class** — AI programs live or die by evaluation, so "measure
   behaviour against examples" belongs *in the language* (assertions over distributions / expected
   behaviour), not in a notebook.

## Design invariants (bet on these; they outlive any specific model/API)

Non-determinism · partiality (refusal / malformed output) · cost & latency · structured output ·
the need for evaluation. Design around these; do **not** encode 2026's API shapes — models change
monthly, the invariants don't.

## Syntax stance — reverse the terseness

Counterintuitive but important: this is a language you write **with** an AI, and LLMs are most
reliable on **conventional, familiar** syntax (in-distribution). Churing's terse sigils
(`~`/`|>`/`@`, prefix `eq`/`gt`) are out-of-distribution → more model errors. So: **conventionalize
the surface.** Keep the pleasant low-ceremony bits (auto-loaded stdlib, auto-print) — those help an
AI (no import bookkeeping) — but drop terseness-for-terseness.

## First experiment (smallest thing that tells us if the idea is real)

In the **interpreter**, with a **deterministic mock backend** (no API keys, offline, reproducible
tests — essential for developing *semantics*):
1. Add `ask` as an `AI`-effect primitive (mock returns canned/typed responses).
2. Implement **type-directed structured output** for one concrete type (e.g. `ask` for a
   `{label: String, score: Float}` → runtime parses/validates/retries).
3. Write **one small program** that is *clearly better* in this style than the bolted-on-library
   version — e.g. a classify-or-extract pipeline with typed retry + an inline eval over examples.
4. **Success criterion:** does the typed/effectful version make the program clearer, safer, more
   composable? If yes → expand. If no → rethink the thesis before building more.

## Open forks (decide deliberately, don't drift)

1. **Vehicle:** extend Churing's interpreter (fast start, already exists) **vs** an eDSL in
   OCaml/Haskell (rides the Python-less FP ecosystem). → *Start in Churing's interpreter; revisit
   only if model-SDK/ecosystem gravity becomes the bottleneck.*
2. **Syntax:** keep Churing's surface vs conventionalize. → *Lean conventionalize (AI-writability).*
3. **Effect system depth:** monadic `AI` (simple, do-notation-ish) vs algebraic effects/handlers
   (more powerful, more design). → *Start monadic; revisit if handlers earn their keep.*
4. **Distribution representation:** `[Scored a]` vs a dedicated `Dist a`. → *Decide at experiment time.*

## Parked (not deleted)

Native compile, ML/tensor stack, CUDA, web, AWS — all committed in git, all genuine learning
artifacts, **no forward investment**. GitHub issues closed with a pointer to #122; reopen if the
direction ever changes. See `docs/native-compile-roadmap.md`.

## How to resume

Read this note. The next action is the **mock-`ask` interpreter experiment** above. The relevant
implementation files: `src/eval.ml` (add the `AI` effect + `ask` primitive + mock backend),
`src/infer.ml` (type the `AI` effect), `src/parser.ml` (surface for `ask`), `src/test/` (an
example program + eval). Keep it interpreter-only.
