# Test JSON parsing and generation (#50)

# toJson — primitives
assert (eq (toJson 42) "42")
assert (eq (toJson "hello") "\"hello\"")
assert (eq (toJson true) "true")
assert (eq (toJson false) "false")
assert (eq (toJson nil) "null")

# toJson — list
assert (eq (toJson [1, 2, 3]) "[1,2,3]")

# toJson — dict
@data {name: "Alice", age: 30}
@json (toJson data)
assert (contains json "\"name\":\"Alice\"")
assert (contains json "\"age\":30")

# toJson — nested
@nested {users: [{name: "Bob"}, {name: "Eve"}], count: 2}
@json2 (toJson nested)
assert (contains json2 "\"users\"")
assert (contains json2 "\"Bob\"")

# fromJson — primitives
assert (eq (fromJson "42") 42)
assert (eq (fromJson "3.14") 3.14)
assert (eq (fromJson "\"hello\"") "hello")
assert (eq (fromJson "true") true)
assert (eq (fromJson "false") false)
assert (eq (fromJson "null") nil)

# fromJson — array
@arr (fromJson "[1, 2, 3]")
assert (eq arr [1, 2, 3])

# fromJson — object
@obj (fromJson "{\"x\": 1, \"y\": 2}")
assert (eq (get obj "x") 1)
assert (eq (get obj "y") 2)

# fromJson — nested
@deep (fromJson "{\"data\": [1, 2], \"ok\": true}")
assert (eq (get deep "data") [1, 2])
assert (eq (get deep "ok") true)

# Roundtrip: toJson → fromJson
@original {name: "test", scores: [90, 85]}
@roundtrip (fromJson (toJson original))
assert (eq (get roundtrip "name") "test")
assert (eq (get roundtrip "scores") [90, 85])

# Error handling — monadic (attempt)
@bad (unwrapOr "parse error" (attempt (|>_. fromJson "invalid")))
assert (eq bad "parse error")
