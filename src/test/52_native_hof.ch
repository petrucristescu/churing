# Test: polymorphic lambdas and HOF in native compile pipeline
# Uses interpreter-compatible API (nth list int, sum, len)

@doubled (map (|>x. x * 2.0) [1.0, 2.0, 3.0])
assert (eq (sum doubled) 12.0)
assert (eq (len doubled) 3.0)
assert (eq (head doubled) 2.0)

@evens (filter (|>x. gt x 2.0) [1.0, 2.0, 3.0, 4.0, 5.0])
assert (eq (len evens) 3.0)
assert (eq (sum evens) 12.0)
assert (eq (head evens) 3.0)

@total (foldl (|>acc. |>x. acc + x) 0.0 [1.0, 2.0, 3.0, 4.0])
assert (eq total 10.0)

@squares (map (|>x. x * x) [1.0, 2.0, 3.0, 4.0, 5.0])
assert (eq (sum squares) 55.0)

# Lambda capturing a variable
@scale 3.0
@scaled (map (|>x. x * scale) [1.0, 2.0, 3.0])
assert (eq (sum scaled) 18.0)
assert (eq (head scaled) 3.0)
