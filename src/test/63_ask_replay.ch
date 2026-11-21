# Replay mode: answers come from a recorded cache, fully offline. The n-th identical
# (model, prompt) call gets the n-th recorded answer; a cache miss is err.

assert (eq (unwrapOr "" (ask "Say hi")) "hi")

@try1 (ask "Classify: great!")
@try2 (ask "Classify: great!")
assert (eq (unwrapOr "" try1) "garbage")
assert (eq (get (fromJson (unwrapOr "{}" try2)) "label") "pos")

assert (isErr (ask "Classify: great!"))
assert (isErr (ask "Never recorded"))
