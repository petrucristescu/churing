# Churing — MCP-Native Semantic Effects (exploration & research)

> **Status: EXPLORATION / research note** (2026-09-14). Not yet a committed direction.
> Extends the AI-native thesis in `docs/ai-native-design.md` (epic #122). Seeded by
> `churing_ai_native_language_analysis.pdf` (a vision doc), then sharpened in discussion.
> Purpose: durable resume point so the strategic thinking isn't lost.

## TL;DR

Explore Churing as a language whose **type system natively knows the categories through
which programs touch the world** — `Db`, `Queue`, `Net`, `Clock`, `AI` — reasons about their
**guarantees**, and keeps their **implementations replaceable via MCP**. AI is not the only
blessed effect; it is one member (the *stochastic* one) of a blessed pantheon of world-effects.

Governing rule that keeps the core small and avoids the "hardcode Postgres" trap:

> **Bless the *categories and their laws* into the compiler. Keep the *implementations* in MCP.**

- **Native in the compiler** (finite ~6–8, stable — matches the design invariants, not 2026 APIs):
  the effect *kinds*, their operation shapes, their **guarantee lattice** (transactional,
  at-least-once, ordered, read-only, idempotent), the **cross-effect consistency rules**, and the
  natural surface syntax.
- **Never in the compiler**: Postgres, Kafka, SQS, Ollama, OpenAI. Those are **MCP bindings**
  chosen at deploy time.

So the compiler understands *what a queue is* (semantics + guarantees); MCP supplies *which queue*.

## How we got here (thesis evolution)

1. Started from `docs/ai-native-design.md`: narrow thesis — `ask` as a typed `AI` effect; keep the
   deterministic core small; the only novel bet is the stochastic edge.
2. The PDF pushed broader: Database/Queue/Network as language-level semantic effects, replaceable
   implementations, capabilities, guarantees, IaC generation. Much of it (DB/Queue-specific
   features, IaC) is the "re-tread mature tools" trap already rejected for the native/ML/AWS work.
3. Key user reframe: **don't relegate resources to a userland effect library** (that's just
   Koka/Unison/Haskell — "what other languages already do"). Put the effect taxonomy **at the
   language level** — the compiler should *know* what a queue/db is, like it knows `Int`.
4. Key architectural insight: **MCP sits between the language and the concrete backend.** MCP is
   the universal driver/binding layer. The language learns "call a tool on a server," never a
   product. This is what makes a *closed blessed taxonomy* buildable without hardcoding products.

The result is a deliberate **thesis pivot**: from "AI-native FP language" to "a language whose
primitives are the semantic categories of the outside world (AI being one of them)."

## Why language-level, not a library (the crux)

A library (even a macro-rich one) fundamentally cannot do what a blessed compiler primitive can:

- **Reason *across* effects** — only the compiler, knowing both `Db` and `Queue`, can reject a
  function that does a store-write + channel-publish as a non-atomic distributed-transaction hazard.
- **Enforce laws, not document them** — "at-least-once ⇒ idempotent consumer" checked
  program-wide and failing the build, not a doc comment.
- **Native surface with semantic authority** — `from users where users.active` where the checker
  knows `users` is a `Db` resource and routes/optimizes (predicate pushdown).
- **One effect algebra** — Db/Queue/AI tracked in one blessed effect row under uniform rules.

A library gives *zero* of these. That gap is the language.

## What it looks like (illustrative surface — conventionalized, not today's `~`/`@` sigils)

### 1. An effect is "call an MCP tool," typed
```
getUser : Db -> UserId -> {Db} (Option User)
getUser db id =
  call db "get_user" { id: id } as Option User
```
`call` invokes the MCP tool, then parses/validates/retries the JSON result into the type
(same structured-output machinery as `ask` / #124).

### 2. Binding happens once, at the edge — never in the logic
```
app {
  use Db   from mcp "postgres-mcp"   # swap to sqlite-mcp / dynamo-mcp: logic unchanged
  use AI   from mcp "ollama-mcp"
  use Mail from mcp "sendgrid-mcp"
  run handleSignup
}
```
MCP servers are self-describing (JSON-Schema per tool) → introspect at bind time → generate the
callable surface + Churing types. The language is "aware of the DB" without being taught a DB.

### 3. Natural surface, routed through MCP
```
activeUsers : {Db} [User]
activeUsers = from db.users where users.active
# ⟶ call db "query" { table: "users", where: { active: true } } ⟶ SQL ⟶ [User]
```

### 4. AI + DB compose; capabilities are the safety story
```
# Handed Db and AI, NOT Mail — type ({Db,AI}) and runtime both forbid sending mail.
summarizeOrders : Db -> AI -> UserId -> {Db, AI} Report
summarizeOrders db ai userId =
  let orders  = from db.orders where orders.userId == userId
  let summary = ask ai "Summarize these orders: {orders}" as Summary
  Report { user: userId, summary: summary }
```

### 5. The two directions become one: `ask` with tools = MCP
```
answer : AI -> Db -> Question -> {AI, Db} Answer
answer ai db q =
  ask ai q using [ tool db.orders, tool db.customers ] as Answer
  # model may call ONLY those MCP tools mid-inference; the toolbelt IS the capability set
```

### 6. Compiler natively reasons (the thing a library can't emit)
```
createOrder : {Db, Queue} ()
createOrder =
  db.save(order)
  queue.publish(OrderCreated order.id)
# error: consistency hazard — a Db write and a Queue publish span two independent systems
#        and cannot be atomic. declare a strategy:  consistent eventual { … }  (lowers to outbox)
```

## Extensibility model (the two axes of "extend")

- **Axis 1 — same effect, different backend** (free, one line): `Db` on Postgres vs SQLite vs
  DynamoDB vs in-memory mock is only a binding swap. Logic untouched.
- **Axis 2 — a new effect *kind*** (Queue, VectorStore, GraphDb, Blob): differs in *shape*, not
  *mechanism*. In the blessed-taxonomy model this is a **language decision** (add to the blessed
  set + its laws), NOT a library import — that is the deliberate, expensive, novel choice.
- **Blessed core + open frontier**: curated effects get full native reasoning; a generic raw-MCP
  `call` escape hatch covers the long tail *without* the compiler's guarantees.

Important limit: MCP gives uniform **reach**, NOT free **cross-effect atomicity**. Two MCP servers
are two independent systems. Distributed transactions stay explicit (outbox/saga/declared eventual);
the language must refuse to pretend otherwise.

## Prior art (almost every piece exists; the *fusion* is unclaimed)

- **Language-level resources, swappable backend (Db/Queue half):**
  - **Winglang** — closest existing thesis: `cloud.Queue`/`Bucket`/`Counter` as language
    primitives, one queue backed by SQS/Azure/RabbitMQ; preflight (infra) vs inflight (runtime).
    **Study its lowest-common-denominator struggles — that's our cautionary tale.**
  - **Ballerina** — network services/data first-class in the language.
  - **Dark (Darklang)** — "deployless" backend as language; failed as a *business* (adoption, not
    tech). Cautionary tale for adoption.
  - **Unison** — *abilities* (algebraic effects) + content-addressed distribution — but abilities
    are user-definable, i.e. the *generic effect* camp we explicitly reject.
- **Native query surface → backend:** **LINQ** (C#), **Links** (Wadler), **Ur/Web**. Solved for
  decades; also where the leaks show.
- **Capabilities as a language primitive:** the **E language**, **object-capability** model,
  **Pony** reference capabilities. Our "AI can't touch what it wasn't handed" is textbook ocap.
- **AI as a first-class typed construct:**
  - **BAML** (typed DSL, structured output, retries → TS/Py/Go) and **DSPy** (typed signatures,
    compiler optimizes prompts) — but *frameworks/DSLs*, not languages with an effect system.
  - **Jac / "Meaning-Typed Programming"** (`by llm` operator) — closest *language-level* NL↔code
    bridge; most relevant to the "sound natural" goal. (arXiv 2405.08965)
  - **λ_A** — typed lambda calculus for LLM agent composition; LLM calls as IO effect. (arXiv
    2604.11767)
- **Universal tool binding:** **MCP** itself (Nov 2024; 2025-11 spec added async "Tasks";
  capability negotiation is first-class).

**Nobody has combined:** a *closed, blessed taxonomy of world-effects with AI as a peer*, bound at
runtime through *MCP as the universal driver*, with the *compiler natively reasoning about
cross-effect guarantees*, aimed at the *NL↔code bridge*. The pieces exist; the integration is open.

## Drawbacks / risks (ranked — performance is NOT #1)

1. **Lowest-common-denominator / leaky abstraction (the real killer).** A *closed blessed* `Queue`
   must flatten Kafka/SQS/RabbitMQ (partitions/offsets/consumer-groups vs visibility-timeouts vs
   exchanges) into either the intersection (lose Kafka's power) or a giant leaky interface. Same for
   `Db` (SQL vs DynamoDB vs graph). Classic ORM trap; **semantic, not fixable by engineering.**
   Winglang lives this. **Disprove this first.**
2. **Adoption.** BAML/DSPy won by being *libraries in Python* (meet users where they are). A new
   language + closed effect model + MCP dependency asks people to leave their ecosystem. Wing,
   Ballerina, Dark all technically interesting, none mainstream; **Dark died here.** Keep the
   interpreter tiny and demo-driven; don't bet on ecosystem gravity.
3. **Compiler burden.** Every blessed effect is permanent compiler surface (type rules, guarantee
   lattice, cross-effect laws) to maintain forever — the tax you pay for native reasoning vs a
   write-once generic effect system.
4. **Guarantees are trusted, not proven.** The compiler trusts the binding annotation
   (`Queue @ at_least_once`); the real guarantee lives in the MCP server/backend config it can't
   verify. It's **a linter with teeth backed by declared trust**, not a theorem. Don't oversell.
5. **The AI effect resists the reasoning that's the point.** Db/Queue have algebraic laws; the AI
   effect is stochastic and has none. Compiler-native reasoning is strongest for the *boring*
   effects and weakest exactly where the novelty is. Only handle on the AI edge is runtime
   eval/observability, not compile-time proof. **Disprove this second.**
6. **Performance (real, bounded, fixable).** MCP-in-the-middle = JSON-RPC over stdio/HTTP:
   serialization + IPC/network hop per op vs a native driver's binary protocol + pooling. "Thin"
   fetch-then-filter routing is catastrophically chatty (N+1 over a protocol boundary). Mitigations:
   predicate pushdown, batching, connection reuse, in-process bindings for hot effects (escape
   hatch). A tax you engineer down, not a wall — hence below the semantic risks.
7. **Debuggability / opacity.** `from … where` lowering through effect → MCP tool → server → SQL
   puts failures/slow queries four layers from source, across a protocol boundary.

Strength (not a drawback): offline **mock-MCP** discipline makes testing/reproducibility *better*
than most effectful systems — same discipline that made the `ask` experiment land.

## Recommended next steps (when this exploration resumes)

1. **Decide the thesis pivot deliberately** — update `docs/ai-native-design.md` if we commit to the
   blessed-world-effect-taxonomy identity (AI as one peer effect). Don't drift into it.
2. **Finish #124** (type-directed structured output for `ask`) — shared machinery with typed MCP
   `call`/tool results.
3. **Stage-1 prototype (smallest thing that proves generality):** in the interpreter, a mock-backed
   `call handle "tool" {args} as Type` primitive + `app { use Db from mcp … }` binding, with the
   *same* primitive backing *two* effect signatures (`Db` and `Queue`) against two mock MCP servers,
   in one offline test program. Demonstrates "one mechanism, many abstractions" end-to-end.
4. **Treat drawbacks #1 (LCD/leaky) and #5 (AI resists reasoning) as the hypotheses to disprove
   first** — not performance.

## Sources

- Winglang: https://github.com/winglang/wing · https://thenewstack.io/if-dev-and-ops-had-a-baby-it-would-be-called-winglang/
- Ballerina: https://ballerina.io/ · https://en.wikipedia.org/wiki/Ballerina_(programming_language)
- Meaning-Typed Programming (Jac): https://arxiv.org/pdf/2405.08965
- λ_A typed lambda calculus for LLM agents: https://arxiv.org/pdf/2604.11767
- DSPy: https://arxiv.org/pdf/2310.03714 · BAML: https://www.btbytes.com/BAML
- MCP one-year / 2025-11 spec: https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/
