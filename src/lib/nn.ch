# Neural network core: forward pass, backpropagation, SGD
# Architecture: 1024 -> 128 -> 64 -> 10
# Uses native FloatArrays for performance

# Xavier initialization scale: sqrt(6 / (fan_in + fan_out))
~xavierScale fanIn,fanOut (sqrt (6.0 / (toFloat (fanIn + fanOut))))

# Create a random matrix (list of FloatArray rows)
~matRandomN rows,cols,scale (match rows
    | 0 -> []
    | _ -> cons (arrayRandom cols scale) (matRandomN (rows - 1) cols scale))

# Initialize network with random weights
~initNetwork dummy (
    @w1 (matRandomN 128 1024 (xavierScale 1024 128))
    @b1 (arrayCreate 128 0.0)
    @w2 (matRandomN 64 128 (xavierScale 128 64))
    @b2 (arrayCreate 64 0.0)
    @w3 (matRandomN 11 64 (xavierScale 64 11))
    @b3 (arrayCreate 11 0.0)
    {w1: w1, b1: b1, w2: w2, b2: b2, w3: w3, b3: b3}
)

# Forward pass: returns dict of all activations for backprop
~forward net,input (
    @w1 (get net "w1")
    @b1 (get net "b1")
    @w2 (get net "w2")
    @b2 (get net "b2")
    @w3 (get net "w3")
    @b3 (get net "b3")

    # Layer 1: input -> hidden1 (relu)
    @z1 (arrayVecAdd (arrayMatVecMul w1 input) b1)
    @a1 (arrayRelu z1)

    # Layer 2: hidden1 -> hidden2 (relu)
    @z2 (arrayVecAdd (arrayMatVecMul w2 a1) b2)
    @a2 (arrayRelu z2)

    # Layer 3: hidden2 -> output (softmax)
    @z3 (arrayVecAdd (arrayMatVecMul w3 a2) b3)
    @a3 (arraySoftmax z3)

    {input: input, z1: z1, a1: a1, z2: z2, a2: a2, z3: z3, output: a3}
)

# Predict: forward pass -> index of highest output
~predict net,input (
    @acts (forward net input)
    arrayArgmax (get acts "output")
)

# One-hot encode using FloatArray
~oneHotA label,size (
    @arr (arrayCreate size 0.0)
    arraySet arr label 1.0
)

# Backpropagation: compute gradients for all layers
~backward net,acts,label (
    @w2 (get net "w2")
    @w3 (get net "w3")
    @input (get acts "input")
    @a1 (get acts "a1")
    @z1 (get acts "z1")
    @a2 (get acts "a2")
    @z2 (get acts "z2")
    @output (get acts "output")

    # Output layer error: output - one_hot(label)
    @target (oneHotA label 11)
    @d3 (arrayVecSub output target)

    # Gradients for layer 3
    @dw3 (arrayOuterProduct d3 a2)
    @db3 d3

    # Hidden layer 2 error: (W3^T * d3) * relu'(z2)
    @w3t (arrayMatTranspose w3)
    @d2pre (arrayMatVecMul w3t d3)
    @d2 (arrayVecMul d2pre (arrayReluDeriv z2))

    # Gradients for layer 2
    @dw2 (arrayOuterProduct d2 a1)
    @db2 d2

    # Hidden layer 1 error: (W2^T * d2) * relu'(z1)
    @w2t (arrayMatTranspose w2)
    @d1pre (arrayMatVecMul w2t d2)
    @d1 (arrayVecMul d1pre (arrayReluDeriv z1))

    # Gradients for layer 1
    @dw1 (arrayOuterProduct d1 input)
    @db1 d1

    {dw1: dw1, db1: db1, dw2: dw2, db2: db2, dw3: dw3, db3: db3}
)

# Update weights with SGD: w = w - lr * gradient
~updateWeights net,grads,lr (
    @nlr (0.0 - lr)
    @w1 (arrayMatAdd (get net "w1") (arrayMatScale nlr (get grads "dw1")))
    @b1 (arrayVecAdd (get net "b1") (arrayVecScale nlr (get grads "db1")))
    @w2 (arrayMatAdd (get net "w2") (arrayMatScale nlr (get grads "dw2")))
    @b2 (arrayVecAdd (get net "b2") (arrayVecScale nlr (get grads "db2")))
    @w3 (arrayMatAdd (get net "w3") (arrayMatScale nlr (get grads "dw3")))
    @b3 (arrayVecAdd (get net "b3") (arrayVecScale nlr (get grads "db3")))
    {w1: w1, b1: b1, w2: w2, b2: b2, w3: w3, b3: b3}
)

# Train on a single example
~trainOne net,input,label,lr (
    @acts (forward net input)
    @grads (backward net acts label)
    updateWeights net grads lr
)

# Train on a list of (input, label) pairs
~trainBatch net,samples,lr (
    foldl (|>net. |>sample. trainOne net (get sample "input") (get sample "label") lr) net samples
)
