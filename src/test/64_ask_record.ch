# Record mode: answers from the backend (here the scripted mock) are written to the
# cache file, so a later run can replay them offline. Errors are not recorded.

assert (eq (unwrapOr "" (ask "Capital of France?")) "Paris")
assert (isOk (ask "Retry me"))

@cache (readFile "test_results/64_ask_record.cache.json")
assert (contains cache "\"prompt\":\"Capital of France?\"")
assert (contains cache "\"response\":\"Paris\"")
assert (contains cache "\"model\":\"rec-model\"")
assert (contains cache "Retry me")
