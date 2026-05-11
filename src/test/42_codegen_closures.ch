# Closures and higher-order functions — interpreter behaviour spec for #92 codegen
# These patterns must compile correctly in the LLVM backend.

# --- Lambda with no capture ---
@double (|>x. x * 2.0)
assert (eq (double 3.0) 6.0)
assert (eq (double 0.5) 1.0)

# --- Lambda with one captured variable ---
@n 10.0
@add_n (|>x. x + n)
assert (eq (add_n 5.0) 15.0)
assert (eq (add_n 0.0) 10.0)

# --- Lambda with multiple captures ---
@a 3.0
@b 4.0
@hyp (|>c. sqrt (a * a + b * b + c * c))
assert (eq (hyp 0.0) 5.0)

# --- @-binding inside named function body ---
~scale_shift x,factor,offset (
    @scaled (x * factor)
    scaled + offset
)
assert (eq (scale_shift 2.0 3.0 1.0) 7.0)
assert (eq (scale_shift 0.0 100.0 5.0) 5.0)

# --- Named function passed to higher-order function ---
~apply f,x  (f x)
~inc x      (x + 1.0)
assert (eq (apply inc 4.0) 5.0)
assert (eq (apply inc 0.0) 1.0)

# --- Closure returned from function (currying via lambda) ---
~make_adder x  (|>y. x + y)
@add5 (make_adder 5.0)
assert (eq (add5 3.0) 8.0)
assert (eq (add5 0.0) 5.0)

# --- Compose: pass two functions, apply in sequence ---
~compose_apply f,g,x  (f (g x))
~square x  (x * x)
~negate x  (0.0 - x)
assert (eq (compose_apply negate square 3.0) (0.0 - 9.0))
assert (eq (compose_apply square negate 3.0) 9.0)

# --- Lambda capturing a let-bound variable inside a function ---
~make_multiplier factor (
    @k (factor * 2.0)
    |>x. x * k
)
@triple (make_multiplier 1.5)
assert (eq (triple 4.0) 12.0)

# --- Nested @-bindings with named function calls ---
~sigmoid_deriv x (
    @s (1.0 / (1.0 + exp (0.0 - x)))
    s * (1.0 - s)
)
@sd0 (sigmoid_deriv 0.0)
assert (gt sd0 0.24)
assert (lt sd0 0.26)
