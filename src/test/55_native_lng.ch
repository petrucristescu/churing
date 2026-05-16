# Test: Lng (Int64) literals in native compile pipeline
@x 1000000000L
assert (eq x 1000000000.0)

@y 42L
assert (eq y 42.0)
