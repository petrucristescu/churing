# Loss functions for neural network

# One-hot encode helper: lifted to top level for native compile
~oneHotBuild label,size,i (match (eq i size)
    | true -> []
    | false -> cons (matchBool (eq i label) 1.0 0.0) (oneHotBuild label size (i + 1)))

# One-hot encode: label -> size -> vector with 1.0 at label index
~oneHot label,size (oneHotBuild label size 0)

# Cross-entropy loss: -log(predicted[label])
# predicted is a probability vector (output of softmax), label is the correct class index
~crossEntropy predicted,label (
    @p (nth predicted label)
    @clipped (matchBool (lt p 0.0000001) 0.0000001 p)
    0.0 - (log clipped)
)
