# Error handling via Result (replaces try/catch)

# safe* combinators — pure failures become Result values
assert (isErr (safeDiv 1.0 0.0))
assert (eq (unwrapOr (0.0 - 1.0) (safeDiv 10.0 2.0)) 5.0)
assert (isErr (safeHead []))
assert (eq (unwrapOr 0 (safeHead [7, 8])) 7)
assert (isErr (safeGet {a: 1} "b"))
assert (eq (unwrapOr 0 (safeGet {a: 1} "a")) 1)

# attempt — catch a thrown runtime error into a Result
@okr (attempt (|>_. 42))
assert (isOk okr)
assert (eq (unwrapOr 0 okr) 42)

@errr (attempt (|>_. head []))
assert (isErr errr)

# recover with a default
assert (eq (unwrapOr "fallback" (attempt (|>_. head []))) "fallback")

# matchResult dispatches on ok / err
assert (eq (matchResult (safeDiv 1.0 0.0) (|>v. "ok") (|>e. "err")) "err")
assert (eq (matchResult (safeDiv 6.0 2.0) (|>v. v) (|>e. (0.0 - 1.0))) 3.0)
