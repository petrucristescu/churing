# 03 — Functions
# Use ~ to define a named function.  Multiple args are comma-separated.

~double x    (x * 2)
~add    a,b  (a + b)
~greet  name (concat "Hello, " name)

str ["double 7 = ", (double 7),
     "  add 3 4 = ", (add 3 4),
     "  greet = ", (greet "world")]
