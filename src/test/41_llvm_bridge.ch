# llvm bridge primitive — interpreter fallback dispatches to OCaml math (same hardware path)

# Direct intrinsic call
assert (eq (llvm "llvm.sqrt.f64" 4.0) 2.0)
assert (eq (llvm "llvm.exp.f64" 0.0) 1.0)
assert (eq (llvm "llvm.fabs.f64" (0.0 - 3.0)) 3.0)

# Two-argument intrinsic
assert (eq (llvm "llvm.pow.f64" 2.0 10.0) 1024.0)
assert (eq (llvm "llvm.minnum.f64" 3.0 7.0) 3.0)
assert (eq (llvm "llvm.maxnum.f64" 3.0 7.0) 7.0)

# Used inside a function definition — this is the intended stdlib pattern
~sqrt x   (llvm "llvm.sqrt.f64" x)
~pow x,y  (llvm "llvm.pow.f64" x y)

assert (eq (sqrt 9.0) 3.0)
assert (eq (pow 2.0 8.0) 256.0)
