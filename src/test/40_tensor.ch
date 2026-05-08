# Test VTensor type — creation, access, immutability, shape, roundtrip

# tensorCreate / tensorGet / tensorRows / tensorCols
@t (tensorCreate 3 2 0.0)
assert (eq (tensorRows t) 3)
assert (eq (tensorCols t) 2)
assert (eq (tensorGet t 0 0) 0.0)
assert (eq (tensorGet t 2 1) 0.0)

# tensorShape
assert (eq (tensorShape t) [3, 2])

# tensorSet returns new tensor (immutable)
@t2 (tensorSet t 1 0 9.9)
assert (eq (tensorGet t2 1 0) 9.9)
assert (eq (tensorGet t  1 0) 0.0)

# tensorVec — [Float] -> Tensor (rows=n, cols=1)
@v (tensorVec [1.0, 2.0, 3.0])
assert (eq (tensorRows v) 3)
assert (eq (tensorCols v) 1)
assert (eq (tensorGet v 0 0) 1.0)
assert (eq (tensorGet v 2 0) 3.0)

# tensorToVec roundtrip
assert (eq (tensorToVec v) [1.0, 2.0, 3.0])

# tensorMat — [[Float]] -> Tensor
@m (tensorMat [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
assert (eq (tensorRows m) 3)
assert (eq (tensorCols m) 2)
assert (eq (tensorGet m 0 0) 1.0)
assert (eq (tensorGet m 0 1) 2.0)
assert (eq (tensorGet m 2 1) 6.0)

# tensorToMat roundtrip
assert (eq (tensorToMat m) [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# tensorRandom — shape and range
@r (tensorRandom 4 4 1.0)
assert (eq (tensorRows r) 4)
assert (eq (tensorCols r) 4)
assert (lt (tensorGet r 0 0) 1.0)
assert (gt (tensorGet r 0 0) (0.0 - 1.0))

# tensorZeros / tensorOnes helpers
@z (tensorZeros 2 3)
assert (eq (tensorGet z 0 0) 0.0)
assert (eq (tensorGet z 1 2) 0.0)

@o (tensorOnes 2 3)
assert (eq (tensorGet o 0 0) 1.0)
assert (eq (tensorGet o 1 2) 1.0)
