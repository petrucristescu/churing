# Test file I/O (Option A — native primitives)

@test_path "/tmp/churing_test_27.txt"
@test_path2 "/tmp/churing_test_27_lines.txt"

# Write a file
assert (writeFile test_path "hello world")

# File exists
assert (fileExists test_path)

# Read it back
assert (eq (readFile test_path) "hello world")

# Append to it
assert (appendFile test_path " more")
assert (eq (readFile test_path) "hello world more")

# Write lines
assert (writeLines test_path2 ["line1", "line2", "line3"])

# Read lines
@lines (readLines test_path2)
assert (eq (len lines) 3)
assert (eq (head lines) "line1")
assert (eq (nth lines 2) "line3")

# Delete files
assert (deleteFile test_path)
assert (not (fileExists test_path))
assert (deleteFile test_path2)

# Error handling — monadic (attempt + unwrapOr)
@result (unwrapOr "caught" (attempt (|>_. readFile "/tmp/nonexistent_churing_file")))
assert (eq result "caught")
