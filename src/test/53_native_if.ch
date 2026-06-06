# Test: if primitive in native compile pipeline

@x 5.0
@y (if (gt x 3.0) 10.0 0.0)
assert (eq y 10.0)

@z (if (lt x 3.0) 10.0 99.0)
assert (eq z 99.0)

# Nested if
@w (if true (if false 1.0 2.0) 3.0)
assert (eq w 2.0)

# if as expression in arithmetic
@r (if (eq x 5.0) 100.0 0.0)
assert (eq r 100.0)
