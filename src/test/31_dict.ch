# Test dictionary/map type (#49) — both approaches

# Option A: Native dict with {} syntax
@user {name: "Alice", age: 30, active: true}

# get
assert (eq (get user "name") "Alice")
assert (eq (get user "age") 30)

# set (returns new dict)
@updated (set user "age" 31)
assert (eq (get updated "age") 31)
assert (eq (get user "age") 30)

# has
assert (has user "name")
assert (not (has user "email"))

# keys / values
assert (eq (len (keys user)) 3)
assert (eq (len (values user)) 3)

# merge
@extra {email: "alice@test.com", role: "admin"}
@merged (merge user extra)
assert (has merged "email")
assert (has merged "name")

# remove
@smaller (remove user "active")
assert (not (has smaller "active"))
assert (has smaller "name")

# entries / fromEntries roundtrip
@pairs (entries user)
@rebuilt (fromEntries pairs)
assert (eq (get rebuilt "name") "Alice")

# Empty dict
@empty {}
assert (eq (len (keys empty)) 0)

# Error handling — monadic (safeGet)
@result (unwrapOr "caught" (safeGet user "missing"))
assert (eq result "caught")

# Option B: Church-encoded dict (association list)
@church_user [["name", "Alice"], ["age", 30]]
assert (eq (assocGet church_user "name") "Alice")
assert (assocHas church_user "name")
assert (not (assocHas church_user "email"))
assert (eq (assocKeys church_user) ["name", "age"])
