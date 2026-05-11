# 06 — Closures and higher-order functions
# |>param. body creates a lambda.  Functions are first-class values.

~makeAdder n  (|>x. x + n)

@add5  (makeAdder 5)
@add10 (makeAdder 10)

@nums  [1, 2, 3, 4, 5]

str ["add5 3 = ", (add5 3),
     "  add10 3 = ", (add10 3),
     "  map add5 = ", (map add5 nums)]
