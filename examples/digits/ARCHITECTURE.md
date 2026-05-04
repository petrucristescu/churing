# Neural Network Digit Recognition — Architecture

A handwritten digit classifier built entirely in Churing. No Python, no external ML libraries — just pure functional programming from pixels to predictions.

## Network Architecture

```
  32×32 PGM Image
  (grayscale)
       │
       │  flatten to 1024 floats (pixel / 255)
       ▼
┌─────────────┐
│ Input Layer  │  1024 neurons (one per pixel)
│  [0.0..1.0] │
└──────┬──────┘
       │
       │  W1: 128×1024 matrix    b1: 128 vector
       │  z1 = W1 · input + b1
       │  a1 = ReLU(z1)
       ▼
┌─────────────┐
│  Hidden 1   │  128 neurons
│    ReLU     │
└──────┬──────┘
       │
       │  W2: 64×128 matrix      b2: 64 vector
       │  z2 = W2 · a1 + b2
       │  a2 = ReLU(z2)
       ▼
┌─────────────┐
│  Hidden 2   │  64 neurons
│    ReLU     │
└──────┬──────┘
       │
       │  W3: 11×64 matrix       b3: 11 vector
       │  z3 = W3 · a2 + b3
       │  a3 = softmax(z3)
       ▼
┌─────────────┐
│ Output Layer │  11 neurons (digits 0-9 + "other")
│   Softmax   │  probabilities sum to 1.0
└──────┬──────┘
       │
       │  argmax(a3)
       ▼
   Predicted
  Digit or "other"
   (0 - 10)
```

### Parameter Count

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| W1    | 128 × 1024 = 131,072 | 128 | 131,200 |
| W2    | 64 × 128 = 8,192 | 64 | 8,256 |
| W3    | 11 × 64 = 704 | 11 | 715 |
| **Total** | | | **140,171** |

## Forward Pass

```
    input (1024)
        │
        ▼
   ┌─────────┐     ┌────┐
   │ matVecMul│────▶│vecA│──▶ z1 (128)
   │  W1×inp  │     │ dd │       │
   └─────────┘     │ +b1│       │
                    └────┘       ▼
                            ┌────────┐
                            │  ReLU  │──▶ a1 (128)
                            │max(0,x)│       │
                            └────────┘       │
                                             ▼
                                        ┌─────────┐     ┌────┐
                                        │ matVecMul│────▶│vecA│──▶ z2 (64)
                                        │  W2×a1   │     │ dd │       │
                                        └─────────┘     │ +b2│       │
                                                         └────┘       ▼
                                                                 ┌────────┐
                                                                 │  ReLU  │──▶ a2 (64)
                                                                 │max(0,x)│       │
                                                                 └────────┘       │
                                                                                  ▼
                                                                             ┌─────────┐     ┌────┐
                                                                             │ matVecMul│────▶│vecA│──▶ z3 (11)
                                                                             │  W3×a2   │     │ dd │       │
                                                                             └─────────┘     │ +b3│       │
                                                                                              └────┘       ▼
                                                                                                      ┌─────────┐
                                                                                                      │ softmax │──▶ output (11)
                                                                                                      │ eˣ/Σeˣ  │
                                                                                                      └─────────┘
```

Each layer computes: **z = W · a_prev + b**, then **a = activation(z)**

The `forward` function returns ALL intermediate values (z1, a1, z2, a2, z3, output) as a dict — these are needed for backpropagation.

## Backpropagation

Gradients flow backwards from the output layer to the input, computing how much each weight contributed to the error.

```
                          ┌──────────────┐
  label ──▶ oneHot(label) │    target    │  [0,0,0,1,0,0,0,0,0,0]  (if label=3)
                          └──────┬───────┘
                                 │
                                 ▼
  output ──────────────▶  d3 = output - target     ◀── Output error (10)
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            │
              ┌──────────┐ ┌──────────┐      │
              │outerProd │ │   db3    │      │
              │ d3 × a2ᵀ │ │  = d3    │      │
              └────┬─────┘ └──────────┘      │
                   ▼                          │
               dW3 (10×64)                    │
                                              │
                                              ▼
  z2 ─▶ reluDeriv(z2) ──┐        d2pre = W3ᵀ · d3
                         ▼              │
                   d2 = d2pre ⊙ relu'(z2)      ◀── Hidden 2 error (64)
                         │
                    ┌────┼────────────┐
                    ▼    ▼            │
              ┌──────────┐ ┌───────┐ │
              │outerProd │ │  db2  │ │
              │ d2 × a1ᵀ │ │ = d2  │ │
              └────┬─────┘ └───────┘ │
                   ▼                  │
               dW2 (64×128)           │
                                      │
                                      ▼
  z1 ─▶ reluDeriv(z1) ──┐        d1pre = W2ᵀ · d2
                         ▼              │
                   d1 = d1pre ⊙ relu'(z1)      ◀── Hidden 1 error (128)
                         │
                    ┌────┼────────────┐
                    ▼    ▼            │
              ┌──────────┐ ┌───────┐ │
              │outerProd │ │  db1  │ │
              │d1 × inpᵀ │ │ = d1  │ │
              └────┬─────┘ └───────┘
                   ▼
               dW1 (128×1024)
```

**Key operations:**
- `⊙` = element-wise multiplication (`vecMul`)
- `×ᵀ` = outer product (`outerProduct`) — produces a gradient matrix from two vectors
- `Wᵀ · d` = transpose then matrix-vector multiply — propagates error backwards

## Weight Update (SGD)

```
  For each weight W and its gradient dW:

      W_new  =  W_old  -  lr × dW

  ┌──────┐     ┌──────┐     ┌──────┐
  │W_old │  -  │  lr  │  ×  │  dW  │  =  W_new
  └──────┘     └──────┘     └──────┘

  Learning rate (lr) = 0.01
  Applied via: matAdd W (matScale (-lr) dW)
```

All weights are immutable — `updateWeights` returns a brand new network dict. No mutation anywhere.

## Activation Functions

```
  ReLU (hidden layers)              Sigmoid (not used, available)
  ┌────────────────────┐            ┌────────────────────┐
  │        ╱           │            │            ___──────│
  │       ╱            │            │          ╱          │
  │──────╱             │            │         ╱           │
  │     0              │            │────────╱            │
  └────────────────────┘            └────────────────────┘
  f(x) = max(0, x)                 f(x) = 1/(1+e⁻ˣ)
  f'(x) = x > 0 ? 1 : 0           f'(x) = f(x)(1-f(x))

  Softmax (output layer)
  ┌────────────────────────────────────────┐
  │  softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ          │
  │                                        │
  │  Input:  [2.0, 1.0, 0.5]              │
  │  Output: [0.57, 0.21, 0.13]  (sum=1)  │
  │                                        │
  │  Numerically stable: subtract max      │
  │  before exp to prevent overflow        │
  └────────────────────────────────────────┘
```

## Weight Initialization (Xavier)

```
  scale = √(6 / (fan_in + fan_out))

  Layer 1: scale = √(6 / (1024 + 128)) = √(0.0052) ≈ 0.072
  Layer 2: scale = √(6 / (128 + 64))   = √(0.0312) ≈ 0.177
  Layer 3: scale = √(6 / (64 + 10))    = √(0.0811) ≈ 0.285

  Each weight: random uniform in [-scale, +scale]
  Biases: initialized to zero
```

Xavier initialization keeps activations and gradients in a reasonable range, preventing vanishing or exploding signals in deep networks.

## Training Data Pipeline

```
  ┌─────────────────────┐
  │  generate_data.ch   │
  │                     │
  │  For each digit 0-9:│
  │  ┌─────────────┐   │
  │  │ digitRects(d)│   │    Digit templates defined as
  │  │ → rectangles │   │    lists of [x1,y1,x2,y2] rects
  │  └──────┬──────┘   │
  │         ▼           │
  │  ┌─────────────┐   │
  │  │ makeImage   │   │    For each pixel i (0..1023):
  │  │ rects→pixels│   │    check if (x,y) falls inside
  │  └──────┬──────┘   │    any rectangle → 1.0 or 0.0
  │         ▼           │
  │  ┌─────────────┐   │
  │  │ shiftImage  │   │    Random shift dx,dy ∈ [-2,+2]
  │  │ dx, dy      │   │    (data augmentation)
  │  └──────┬──────┘   │
  │         ▼           │
  │  ┌─────────────┐   │
  │  │  addNoise   │   │    Random noise ±0.1 per pixel
  │  │  ±0.1       │   │    (data augmentation)
  │  └──────┬──────┘   │
  │         ▼           │
  │  ┌─────────────┐   │
  │  │  writePgm   │   │    Save as 32×32 PGM P2 file
  │  │  32×32      │   │    e.g., data/train/3_007.pgm
  │  └─────────────┘   │
  └─────────────────────┘

  Output: 165 training images (15 per digit/class, including "other")
          55 test images (5 per digit/class)
```

### PGM P2 Format

```
  P2              ← magic number (ASCII grayscale)
  32 32           ← width height
  255             ← max pixel value
  0 0 0 0 ...    ← pixel values (space-separated)
  255 255 0 ...     row by row, left to right, top to bottom
  ...
```

Standard image format — viewable in GIMP, IrfanView, ImageMagick, Firefox.

## Training Loop

```
  ┌─────────────┐
  │ Load PGMs   │    Read all data/train/*.pgm files
  │ from disk   │    Parse label from filename (3_007.pgm → label=3)
  └──────┬──────┘    Normalize pixels to [0.0, 1.0]
         │
         ▼
  ┌─────────────┐
  │ initNetwork │    Random Xavier weights
  │  140k params│    All biases = 0
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────────┐
  │  FOR epoch = 1 to 5:                        │
  │                                              │
  │    FOR each sample (input, label):           │
  │    ┌──────────────────────────────────────┐  │
  │    │  1. forward(net, input)  → acts      │  │
  │    │  2. backward(net, acts, label) → grad│  │
  │    │  3. updateWeights(net, grad, 0.01)   │  │
  │    │     → new_net                        │  │
  │    └──────────────────────────────────────┘  │
  │                                              │
  │    Compute accuracy on training set           │
  │    Print: "Epoch N - accuracy: X"            │
  └──────────────────────┬──────────────────────┘
                         │
                         ▼
  ┌─────────────┐
  │ Save weights│    toJson(net) → weights.json
  │  as JSON    │    (~140k float values serialized)
  └─────────────┘
```

## Inference

```
  ┌──────────────┐
  │ weights.json │    Load saved network
  │  fromJson    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  digit.pgm   │    32×32 grayscale image
  │   readPgm    │    (hand-drawn or generated)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  forward()   │    Run through all 3 layers
  │  → output    │    Get 10 probability scores
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  argmax()    │    Index of highest probability
  │  → digit     │    = predicted digit
  └──────────────┘

  Example output:
    Image: data/test/7_002.pgm
    Predicted: digit 7
    Confidence: [0.01, 0.00, 0.02, 0.01, 0.00, 0.01, 0.03, 0.89, 0.01, 0.02, 0.00]
```

## Data Representation

Everything is pure functional — no mutable state anywhere.

| Concept | Churing Representation |
|---------|----------------------|
| Image | `[Float]` — flat list of 1024 floats |
| Network | `Dict {w1, b1, w2, b2, w3, b3}` |
| Weight matrix | `[[Float]]` — list of row vectors |
| Bias vector | `[Float]` |
| Training sample | `Dict {input: [Float], label: Int}` |
| Activations | `Dict {input, z1, a1, z2, a2, z3, output}` |
| Gradients | `Dict {dw1, db1, dw2, db2, dw3, db3}` |
| Saved model | JSON file (via `toJson`/`fromJson`) |

## Library Dependency Graph

```
  operators  math  string  list  dict  json  io
      │        │      │      │     │     │    │
      └────────┴──────┴──────┴─────┴─────┴────┘
                          │
                       ┌──┴──┐
                       │     │
                    vector  pgm
                       │
                    matrix
                       │
                  activations
                       │
                     loss
                       │
                      nn
                    ╱    ╲
               train.ch  predict.ch
```

## Running

```bash
# Generate training data (150 train + 50 test PGM images)
./run.sh examples/digits/generate_data.ch

# Train the network (5 epochs)
./run.sh examples/digits/train.ch

# Predict a digit from a PGM image
./run.sh examples/digits/predict.ch
```
