# Test: File I/O in native compile pipeline

@content "hello native file io"
@path "/tmp/churing_test_59.txt"

# write and read back
@_ (writeFile path content)
@read_back (readFile path)
assert (eq read_back content)

# fileExists
assert (fileExists path)
assert (not (fileExists "/tmp/churing_nonexistent_xyz.txt"))

# appendFile
@_ (appendFile path " appended")
@read2 (readFile path)
assert (eq read2 "hello native file io appended")

# deleteFile
@_ (deleteFile path)
assert (not (fileExists path))
