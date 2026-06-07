# Real LLM test — requires Ollama (run via ./run-tests-llm.sh). Skipped by the
# regular suite. Assertions are loose because LLM output is non-deterministic:
# this proves the `ask` pipeline end-to-end (request -> model -> Result),
# not the model's specific answer.

@r (ask "Reply with one short word.")

assert (isOk r)
assert (gt (length (unwrapOr "" r)) 0)

concat "model replied: " (unwrapOr "" r)
