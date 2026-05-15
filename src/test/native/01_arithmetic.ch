~double x (x * 2.0)
~factorial n (match n | 0 -> 1.0 | x -> x * factorial (x - 1.0))
~sum_acc xs,acc (match xs | [] -> acc | h :: t -> sum_acc t (acc + h))

@x (double 5.0)
assert (eq x 10.0)
assert (eq (factorial 5.0) 120.0)
assert (eq (sum_acc [1.0, 2.0, 3.0, 4.0] 0.0) 10.0)
