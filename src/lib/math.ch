# Math Library
# sqrt, sin, cos, tan, asin, acos, atan, exp, log, tanh, pow, min, max,
# floor, ceil, round, abs are native OCaml primitives (registered in eval.ml).
# This file adds constants and pure-Churing helpers on top.

@pi 3.14159265358979323846
@e 2.71828182845904523536

~square x (x * x)
~cube x (x * x * x)
~clamp lo,hi,x (min hi (max lo x))
~lerp a,b,t (a + t * (b - a))
