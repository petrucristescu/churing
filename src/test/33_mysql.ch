# Test MySQL driver — requires running MySQL (use run-tests-db.sh)

# Connect using environment variables
@conn (mysqlConnect 0)

# Query all users (seeded in init.sql)
@users (mysqlQuery conn "SELECT * FROM users ORDER BY id")
assert (eq (len users) 3)
assert (eq (get (head users) "name") "Alice")
assert (eq (get (nth users 1) "name") "Bob")
assert (eq (get (nth users 2) "age") 35)

# Query with WHERE clause
@alice (mysqlQuery conn "SELECT * FROM users WHERE name = 'Alice'")
assert (eq (len alice) 1)
assert (eq (get (head alice) "email") "alice@test.com")

# Query products
@products (mysqlQuery conn "SELECT * FROM products ORDER BY price")
assert (eq (len products) 3)
assert (eq (get (head products) "name") "Doohickey")

# Insert a new row
assert (mysqlExec conn "INSERT INTO users (name, age, email) VALUES ('Eve', 22, 'eve@test.com')")

# Verify insert
@eve (mysqlQuery conn "SELECT * FROM users WHERE name = 'Eve'")
assert (eq (len eve) 1)
assert (eq (get (head eve) "age") 22)

# Update a row
assert (mysqlExec conn "UPDATE users SET age = 23 WHERE name = 'Eve'")
@updated (mysqlQuery conn "SELECT age FROM users WHERE name = 'Eve'")
assert (eq (get (head updated) "age") 23)

# Delete a row
assert (mysqlExec conn "DELETE FROM users WHERE name = 'Eve'")
@after_delete (mysqlQuery conn "SELECT * FROM users WHERE name = 'Eve'")
assert (eq (len after_delete) 0)

# NULL handling
@charlie (mysqlQuery conn "SELECT email FROM users WHERE name = 'Charlie'")
assert (eq (get (head charlie) "email") nil)

# Error handling — monadic (attempt)
@bad (unwrapOr "caught" (attempt (|>_. mysqlQuery conn "SELECT * FROM nonexistent_table")))
assert (eq bad "caught")

# Helper: mysqlFindOne
@one (mysqlFindOne conn "SELECT * FROM users WHERE name = 'Bob'")
assert (eq (get one "age") 25)

# Helper: mysqlFind
@found (mysqlFind conn "users" "name" "Alice")
assert (eq (len found) 1)
assert (eq (get (head found) "age") 30)

# Close connection (no-op but good practice)
assert (mysqlClose conn)
