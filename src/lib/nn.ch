# Neural network core: forward pass, backpropagation, SGD
# Architecture: 1024 -> 128 -> 64 -> 11
# Pure Churing — uses vector.ch, matrix.ch, activations.ch, loss.ch

import "vector"
import "matrix"
import "activations"
import "loss"

# Xavier initialization scale: sqrt(6 / (fan_in + fan_out))
~xavierScale fanIn,fanOut (sqrt (6.0 / (toFloat (fanIn + fanOut))))

# Initialize network with random weights
~initNetwork dummy (
    @w1 (matRandom 128 1024 (xavierScale 1024 128))
    @b1 (vecZeros 128)
    @w2 (matRandom 64 128 (xavierScale 128 64))
    @b2 (vecZeros 64)
    @w3 (matRandom 11 64 (xavierScale 64 11))
    @b3 (vecZeros 11)
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
    @z1 (vecAdd (matVecMul w1 input) b1)
    @a1 (map relu z1)

    # Layer 2: hidden1 -> hidden2 (relu)
    @z2 (vecAdd (matVecMul w2 a1) b2)
    @a2 (map relu z2)

    # Layer 3: hidden2 -> output (softmax)
    @z3 (vecAdd (matVecMul w3 a2) b3)
    @a3 (softmax z3)

    {input: input, z1: z1, a1: a1, z2: z2, a2: a2, z3: z3, output: a3}
)

# Predict: forward pass -> index of highest output
~predict net,input (
    @acts (forward net input)
    argmax (get acts "output")
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
    @target (oneHot label 11)
    @d3 (vecSub output target)

    # Gradients for layer 3
    @dw3 (outerProduct d3 a2)
    @db3 d3

    # Hidden layer 2 error: (W3^T * d3) * relu'(z2)
    @w3t (matTranspose w3)
    @d2pre (matVecMul w3t d3)
    @d2 (vecMul d2pre (map reluDeriv z2))

    # Gradients for layer 2
    @dw2 (outerProduct d2 a1)
    @db2 d2

    # Hidden layer 1 error: (W2^T * d2) * relu'(z1)
    @w2t (matTranspose w2)
    @d1pre (matVecMul w2t d2)
    @d1 (vecMul d1pre (map reluDeriv z1))

    # Gradients for layer 1
    @dw1 (outerProduct d1 input)
    @db1 d1

    {dw1: dw1, db1: db1, dw2: dw2, db2: db2, dw3: dw3, db3: db3}
)

# Update weights with SGD: w = w - lr * gradient
~updateWeights net,grads,lr (
    @nlr (0.0 - lr)
    @w1 (matAdd (get net "w1") (matScale nlr (get grads "dw1")))
    @b1 (vecAdd (get net "b1") (vecScale nlr (get grads "db1")))
    @w2 (matAdd (get net "w2") (matScale nlr (get grads "dw2")))
    @b2 (vecAdd (get net "b2") (vecScale nlr (get grads "db2")))
    @w3 (matAdd (get net "w3") (matScale nlr (get grads "dw3")))
    @b3 (vecAdd (get net "b3") (vecScale nlr (get grads "db3")))
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
