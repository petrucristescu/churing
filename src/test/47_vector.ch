# Test: matchList, matchBool, vector ops in native compile

# matchBool
assert (eq (matchBool true 1.0 0.0) 1.0)
assert (eq (matchBool false 1.0 0.0) 0.0)

# matchList
@xs [1.0, 2.0, 3.0]
@result (matchList xs (|>_. 0.0) (|>h. |>t. h))
assert (eq result 1.0)

@result2 (matchList [] (|>_. 99.0) (|>h. |>t. h))
assert (eq result2 99.0)

# vecAdd
@a [1.0, 2.0, 3.0]
@b [4.0, 5.0, 6.0]
@c (vecAdd a b)
assert (eq (head c) 5.0)
assert (eq (sum c) 21.0)

# vecDot
assert (eq (vecDot a b) 32.0)

# vecScale
@scaled (vecScale 2.0 a)
assert (eq (head scaled) 2.0)
assert (eq (sum scaled) 12.0)

# vecZeros
@zeros (vecZeros 3)
assert (eq (sum zeros) 0.0)
assert (eq (len zeros) 3.0)

# argmax
@scores [0.1, 0.8, 0.3, 0.2]
assert (eq (argmax scores) 1.0)
