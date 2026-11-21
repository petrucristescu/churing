# Scripted mock (CHURING_ASK_MOCK_FILE, see 61_ask_mock_script.ch.env):
# a JSON list gives answers in call order, an {"error": ...} answer becomes err,
# a non-string answer comes back as JSON text, and running out of answers is err.

@a (ask "anything")
assert (isOk a)
assert (eq (unwrapOr "" a) "first")

@b (ask "anything")
assert (isErr b)
assert (eq (matchResult b (|>v. v) (|>e. e)) "rate limited")

@c (ask "classify this")
assert (eq (get (fromJson (unwrapOr "{}" c)) "label") "pos")

@d (ask "one too many")
assert (isErr d)
