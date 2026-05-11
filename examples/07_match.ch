# 07 — Pattern matching
# match works on values, lists, booleans, and cons cells (h :: t).

~describe x
  (match x
    | 0 -> "zero"
    | 1 -> "one"
    | _ -> "many")

~first xs
  (match xs
    | []     -> "empty"
    | h :: _ -> str ["head=", h])

str ["0: ", (describe 0),
     "  2: ", (describe 2),
     "  []: ", (first []),
     "  [7,8]: ", (first [7, 8])]
