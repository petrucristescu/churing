# Test: env, envOr, print strings in native compile

# print a string (uses %s format)
print "hello from native"

# envOr with missing var returns default
@host (envOr "NONEXISTENT_VAR_XYZ" "default_host")
assert (eq host "default_host")

# toString
@s (toString 42.0)
assert (eq s "42")

@s2 (toString 3.14)
assert (eq s2 "3.14")
