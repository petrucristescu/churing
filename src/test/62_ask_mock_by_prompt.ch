# Scripted mock keyed by prompt: exact prompt wins, "*" is the fallback, and a list
# gives successive answers for repeated calls (this is how retries are tested).

assert (eq (unwrapOr "" (ask "Capital of France?")) "Paris")
assert (eq (unwrapOr "" (ask "Something else")) "fallback")

@first (ask "Retry me")
@second (ask "Retry me")
@third (ask "Retry me")
assert (eq (unwrapOr "" first) "not json")
assert (isErr (attempt (|>_. fromJson (unwrapOr "" first))))
assert (eq (get (fromJson (unwrapOr "" second)) "label") "neg")
assert (eq (unwrapOr "" third) (unwrapOr "" second))
