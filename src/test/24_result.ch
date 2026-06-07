# Test Church-encoded Result type (Option B — library)

# Create Ok and Err values
@success (ok 42)
@failure (err "something went wrong")

# matchResult — eliminate Result
assert (eq (matchResult success (|>v. v + 1) (|>e. 0)) 43)
assert (eq (matchResult failure (|>v. v) (|>e. e)) "something went wrong")

# isOk / isErr
assert (isOk success)
assert (not (isOk failure))
assert (isErr failure)
assert (not (isErr success))

# unwrapOr — get value or default
assert (eq (unwrapOr 0 success) 42)
assert (eq (unwrapOr 0 failure) 0)

# mapResult — transform Ok value
@doubled (mapResult (|>v. v * 2) success)
assert (eq (unwrapOr 0 doubled) 84)

# mapResult on Err passes through
@still_err (mapResult (|>v. v * 2) failure)
assert (isErr still_err)

# bindResult — chain computations
~safeDivide x,y (match (eq y 0) | true -> err "divide by zero" | _ -> ok (x / y))
@chained (bindResult (|>v. safeDivide v 2) (ok 10))
assert (eq (unwrapOr 0 chained) 5)

@chained_err (bindResult (|>v. safeDivide v 0) (ok 10))
assert (isErr chained_err)
