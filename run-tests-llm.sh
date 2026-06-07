#!/bin/bash
# Run the LLM integration test against a local Ollama + a small model.
# Mirrors run-tests-db.sh. Proves the `ask` primitive end-to-end.
#
# Usage:
#   ./run-tests-llm.sh                 # default model qwen2.5:0.5b
#   OLLAMA_MODEL=smollm2:135m ./run-tests-llm.sh
set -e

MODEL="${OLLAMA_MODEL:-qwen2.5:0.5b}"
NET=churing-net

echo "=== Starting Ollama ==="
docker network create "$NET" 2>/dev/null || true
docker rm -f churing-ollama 2>/dev/null || true
docker run -d --name churing-ollama --network "$NET" ollama/ollama

echo "=== Pulling $MODEL (first run downloads it) ==="
docker exec churing-ollama ollama pull "$MODEL"

echo "=== Rebuilding Churing image ==="
docker build -t churing-test . -q

echo "=== Running LLM test (src/test/llm_ask.ch) ==="
failed=0
MSYS_NO_PATHCONV=1 docker run --rm --user root --network "$NET" \
    -e OPAMROOTISOK=1 \
    -e OLLAMA_HOST=http://churing-ollama:11434 \
    -e OLLAMA_MODEL="$MODEL" \
    -v "$(pwd):/app" \
    churing-test bash -c "
        eval \$(opam env) && dune build src/churing.exe &&
        _build/default/src/churing.exe src/test/llm_ask.ch
    " 2>&1 || failed=1

echo "=== Tearing down Ollama ==="
docker rm -f churing-ollama 2>/dev/null || true

if [ "$failed" -eq 1 ]; then
    echo "LLM test failed!"
    exit 1
else
    echo "LLM test passed!"
fi
