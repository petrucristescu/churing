# Test: string ops in native compile pipeline

@s "hello world"

assert (eq (indexOf s "world") 6.0)
assert (eq (indexOf s "xyz") (0.0 - 1.0))
assert (eq (indexOf s "hello") 0.0)

assert (eq (charAt s 0) "h")
assert (eq (charAt s 4) "o")
assert (eq (charAt s 6) "w")
