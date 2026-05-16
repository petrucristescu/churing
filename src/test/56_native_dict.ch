# Test: Dict support in native compile pipeline

# Basic dict creation and get
@d {x: 1.0, y: 2.0, z: 3.0}
assert (eq (get d "x") 1.0)
assert (eq (get d "y") 2.0)
assert (eq (get d "z") 3.0)

# Dict with arithmetic on values
@sum ((get d "x") + (get d "y") + (get d "z"))
assert (eq sum 6.0)

# Dict with list values
@items [10.0, 20.0, 30.0]
@d2 {nums: items, count: 3.0}
assert (eq (head (get d2 "nums")) 10.0)
assert (eq (get d2 "count") 3.0)

# has
assert (has d "x")
assert (not (has d "w"))
