# Tail call optimisation (#94)
# Verifies tail-recursive patterns that the LLVM backend marks with `tail call`.
# These run via the interpreter (which uses VTailCall trampoline for TCO).

# --- Self-recursive countdown ---
~countdown n (
    match n
    | 0 -> 0.0
    | x -> countdown (x - 1)
)
assert (eq (countdown 0) 0.0)
assert (eq (countdown 1) 0.0)
assert (eq (countdown 100) 0.0)

# --- Accumulator-style sum (classic TCO pattern) ---
~sum_acc xs,acc (
    match xs
    | [] -> acc
    | h :: t -> sum_acc t (acc + h)
)
assert (eq (sum_acc [] 0.0) 0.0)
assert (eq (sum_acc [1.0, 2.0, 3.0] 0.0) 6.0)
assert (eq (sum_acc [10.0, 20.0, 30.0] 5.0) 65.0)

# --- Tail-recursive length ---
~len_acc xs,acc (
    match xs
    | [] -> acc
    | _ :: t -> len_acc t (acc + 1.0)
)
assert (eq (len_acc [] 0.0) 0.0)
assert (eq (len_acc [1.0, 2.0, 3.0, 4.0, 5.0] 0.0) 5.0)

# --- Non-tail use: result of recursive call is consumed ---
~double_sum xs (
    match xs
    | [] -> 0.0
    | h :: t -> h * 2.0 + double_sum t
)
assert (eq (double_sum []) 0.0)
assert (eq (double_sum [1.0, 2.0, 3.0]) 12.0)
