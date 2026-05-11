# 05 — Recursion
# Functions can call themselves.  Tail calls are optimised automatically.

~factorial n
  (match (eq n 0)
    | true  -> 1
    | false -> n * (factorial (n - 1)))

~fib n
  (match (lte n 1)
    | true  -> n
    | false -> (fib (n - 1)) + (fib (n - 2)))

str ["10! = ", (factorial 10), "  fib 10 = ", (fib 10)]
