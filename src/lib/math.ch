# Math Library — hardware-accelerated via LLVM JIT (#91)
# Native OCaml primitives retained for: floor, ceil, round (return Int),
#   abs (polymorphic Int/Long/Float), tan, asin, acos, atan, tanh (not yet tested as intrinsics)

@pi 3.14159265358979323846
@e 2.71828182845904523536

# Float → Float: override OCaml-bound primitives with LLVM intrinsics
~sqrt x  (llvm "llvm.sqrt.f64" x)
~exp x   (llvm "llvm.exp.f64" x)
~log x   (llvm "llvm.log.f64" x)
~sin x   (llvm "llvm.sin.f64" x)
~cos x   (llvm "llvm.cos.f64" x)

# Float → Float → Float
~pow x,y (llvm "llvm.pow.f64" x y)
~min a,b (llvm "llvm.minnum.f64" a b)
~max a,b (llvm "llvm.maxnum.f64" a b)

# Remaining trig/math — libm (linked with -lm)
~tan x   (llvm "tan" x)
~asin x  (llvm "asin" x)
~acos x  (llvm "acos" x)
~atan x  (llvm "atan" x)
~tanh x  (llvm "tanh" x)

# Rounding — LLVM intrinsics
~floor x (llvm "llvm.floor.f64" x)
~ceil x  (llvm "llvm.ceil.f64" x)
~round x (llvm "llvm.round.f64" x)
~abs x   (llvm "llvm.fabs.f64" x)

# Pure Churing helpers
~square x (x * x)
~cube x (x * x * x)
~clamp lo,hi,x (min hi (max lo x))
~lerp a,b,t (a + t * (b - a))
