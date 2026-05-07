# Test native array primitives (arrayCreate, arrayGet, arraySet, arrayLength, arrayFromList, arrayToList)

# arrayCreate, arrayGet, arrayLength
@a (arrayCreate 5 0.0)
assert (eq (arrayLength a) 5)
assert (eq (arrayGet a 0) 0.0)
assert (eq (arrayGet a 4) 0.0)

# arraySet returns new array (immutable)
@b (arraySet a 2 3.14)
assert (eq (arrayGet b 2) 3.14)
assert (eq (arrayGet a 2) 0.0)

# arrayFromList / arrayToList roundtrip
@c (arrayFromList [1.0, 2.0, 3.0])
assert (eq (arrayLength c) 3)
assert (eq (arrayGet c 1) 2.0)
assert (eq (arrayToList c) [1.0, 2.0, 3.0])
