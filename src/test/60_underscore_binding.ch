# Regression: @_ (underscore/discard binding) must not crash the parser.
# Previously `@_ <expr>` followed by another statement caused a parser
# stack overflow because the leading @ token was never consumed.

@_ 99.0
@x 7.0
@_ (x + 1.0)
assert (eq x 7.0)
