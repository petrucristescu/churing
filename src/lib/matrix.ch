# Matrix operations for neural network
# A matrix is a list of row vectors (list of lists of floats)

# Matrix-vector multiply: mat (m x n) * vec (n) -> vec (m)
~matVecMul mat,vec (map (|>row. vecDot row vec) mat)

# Get column i from a matrix
~matCol mat,i (map (|>row. nth row i) mat)

# Transpose a matrix using range + map
~matTranspose mat (matchList mat (|>_. []) (|>row. |>_. map (|>i. matCol mat i) (range 0 (len row))))

# Outer product: vec (m) * vec (n) -> mat (m x n)
~outerProduct a,b (map (|>ai. map (|>bj. ai * bj) b) a)

# Element-wise matrix addition
~matAdd a,b (vecZipWith (|>ra. |>rb. vecAdd ra rb) a b)

# Scalar times matrix
~matScale s,mat (map (|>row. vecScale s row) mat)

# Random matrix rows x cols, values in [-scale, scale]
~matRandom rows,cols,scale (match rows
    | 0 -> []
    | _ -> cons (vecRandom cols scale) (matRandom (rows - 1) cols scale))

# Zero matrix rows x cols
~matZeros rows,cols (match rows
    | 0 -> []
    | _ -> cons (vecZeros cols) (matZeros (rows - 1) cols))

# Matrix-matrix multiply: (m x k) * (k x n) -> (m x n)
~matMul A,B (
    @Bt (matTranspose B)
    map (|>row. map (|>col. vecDot row col) Bt) A)
