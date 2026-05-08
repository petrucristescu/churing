# Tensor — contiguous float array with shape (rows x cols)
# Native primitives: tensorCreate, tensorGet, tensorSet,
#   tensorRows, tensorCols, tensorShape,
#   tensorVec, tensorMat, tensorToVec, tensorToMat, tensorRandom

# Zero tensor
~tensorZeros rows,cols (tensorCreate rows cols 0.0)

# Convert a vector (list or tensor) to a 1-col tensor and back
~tensorOnes rows,cols (tensorCreate rows cols 1.0)
