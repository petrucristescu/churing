# Result type — tagged representation.
#   ok v  = ["ok", v]
#   err e = ["err", e]
# Both the stdlib and runtime primitives produce this exact shape, so fallible
# primitives (readFile, fromJson, ...) can return a Result directly — no exceptions,
# no try/catch. The eliminator uses `match` (not matchBool) so only the chosen
# branch evaluates (lazy in the branches, safe under strict evaluation).

# Constructors
~ok v ["ok", v]
~err e ["err", e]

# Eliminator: matchResult result onOk onErr
~matchResult r,onOk,onErr (match (head r) | "ok" -> onOk (head (tail r)) | _ -> onErr (head (tail r)))

# Map over the Ok value
~mapResult f,r (matchResult r (|>v. ok (f v)) (|>e. err e))

# Chain (flatMap / bind)
~bindResult f,r (matchResult r f (|>e. err e))

# Get the Ok value or a default
~unwrapOr default,r (matchResult r (|>v. v) (|>_. default))

# Predicates
~isOk r (eq (head r) "ok")
~isErr r (eq (head r) "err")

# Safe combinators — turn in-language failures into Result (replaces try/catch
# for the pure cases). Each uses `match` so the failing expression is never
# evaluated on the error branch.
~safeDiv a,b (match (eq b 0.0) | true -> err "division by zero" | _ -> ok (a / b))
~safeHead xs (match xs | [] -> err "head: empty list" | h :: t -> ok h)
~safeTail xs (match xs | [] -> err "tail: empty list" | h :: t -> ok t)
~safeNth n,xs (match (lt n (len xs)) | true -> ok (nth n xs) | _ -> err "index out of range")
~safeGet d,k (match (has d k) | true -> ok (get d k) | _ -> err "missing key")
