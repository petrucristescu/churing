# List operations and pattern matching (#93)
# Documents patterns the LLVM backend compiles: list literals, nil/cons match,
# recursive list functions, integer pattern matching.

# --- List literals ---
@xs [1.0, 2.0, 3.0, 4.0, 5.0]
assert (eq (head xs) 1.0)
assert (eq (tail xs) [2.0, 3.0, 4.0, 5.0])
assert (not (empty xs))
assert (empty [])

# --- Cons constructor ---
@ys (cons 0.0 xs)
assert (eq (head ys) 0.0)
assert (eq (len ys) 6)

# --- Match nil / cons — recursive sum (returns f64) ---
~sum xs (
    match xs
    | [] -> 0.0
    | h :: t -> h + sum t
)
assert (eq (sum []) 0.0)
assert (eq (sum [1.0, 2.0, 3.0]) 6.0)
assert (eq (sum [10.0, 20.0, 30.0, 40.0]) 100.0)

# --- Recursive length via match ---
~mylen xs (
    match xs
    | [] -> 0.0
    | _ :: t -> 1.0 + mylen t
)
assert (eq (mylen []) 0.0)
assert (eq (mylen [1.0, 2.0, 3.0]) 3.0)

# --- Head extraction via cons match ---
~myhead xs (match xs | h :: _ -> h)
assert (eq (myhead [42.0, 1.0, 2.0]) 42.0)

# --- Integer pattern matching (factorial) ---
~fact n (match n
    | 0 -> 1
    | x -> x * fact (x - 1))
assert (eq (fact 0) 1)
assert (eq (fact 5) 120)
assert (eq (fact 10) 3628800)

# --- Wildcard pattern ---
@r1 (match 42.0 | _ -> 1.0)
assert (eq r1 1.0)

# --- Variable binding pattern ---
@r2 (match 7.0 | x -> x * 2.0)
assert (eq r2 14.0)

# --- Boolean pattern matching ---
@r3 (match true | true -> 1.0 | false -> 0.0)
assert (eq r3 1.0)

# --- Integer pattern with fallthrough ---
@r4 (match 3 | 0 -> 0.0 | 1 -> 1.0 | _ -> 99.0)
assert (eq r4 99.0)
