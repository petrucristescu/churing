# 04 — Lists
# Lists support map, filter, foldl, and more from the standard library.

@nums [1, 2, 3, 4, 5]

@doubled (map    (|>x. x * 2)       nums)
@big     (filter (|>x. gt x 3)      nums)
@total   (foldl  (|>acc,x. acc + x) 0 nums)

str ["nums=", nums, "  doubled=", doubled, "  big=", big, "  total=", total]
