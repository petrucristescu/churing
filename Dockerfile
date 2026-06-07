FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (matching GitHub Actions setup)
RUN apt-get update && \
    apt-get install -y bubblewrap wget curl gcc g++ make patch unzip m4 git ca-certificates bzip2 default-mysql-client && \
    rm -rf /var/lib/apt/lists/*

# Install opam 2.1.5 (same version as CI)
RUN wget -q https://github.com/ocaml/opam/releases/download/2.1.5/opam-2.1.5-x86_64-linux -O /usr/local/bin/opam && \
    chmod +x /usr/local/bin/opam

# Create a non-root user for builds and test runs
RUN groupadd --system appuser && \
    useradd --system --create-home --gid appuser appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

# Initialize opam with OCaml 5.1.1 (same as CI)
# Set OPAMROOT explicitly so opam works even when HOME is overridden at runtime
ENV OPAMROOT=/home/appuser/.opam
RUN opam init --disable-sandboxing --no-setup --compiler=ocaml-base-compiler.5.1.1 -y && \
    opam update && \
    opam install dune alcotest -y && \
    chmod -R a+rX /home/appuser/.opam

# Make opam root accessible to any UID (for --user overrides in run-tests.sh)
USER root
RUN chmod a+rx /home/appuser

COPY <<'EOF' /usr/local/bin/run-tests-in-docker.sh
#!/usr/bin/env bash
set -e

eval "$(opam env)"
dune build src/churing.exe

echo "=== OCaml unit tests ==="
dune runtest 2>&1 || { echo "Unit tests failed!"; exit 1; }

echo "=== Churing integration tests ==="
mkdir -p test_results

failed=0
for test_file in src/test/*.ch; do
    test_name=$(basename "$test_file" .ch)

    # Skip database tests (require external services)
    case "$test_name" in *_mysql*|*_dynamo*|*llm*) echo "SKIP: $test_name (external service)"; continue;; esac

    expected_output="src/test/${test_name}.ch.out"

    _build/default/src/churing.exe "$test_file" > "test_results/${test_name}.out" 2>&1
    exit_code=$?

    if [ -f "$expected_output" ]; then
        if ! diff -u "test_results/${test_name}.out" "$expected_output"; then
            echo "FAIL: $test_name"
            failed=1
        else
            echo "PASS: $test_name"
        fi
    elif [ "$exit_code" -ne 0 ]; then
        echo "FAIL: $test_name (exit code $exit_code)"
        cat "test_results/${test_name}.out"
        failed=1
    else
        echo "PASS: $test_name"
    fi
done

if [ "$failed" -eq 1 ]; then
    echo "Some tests failed!"
    exit 1
else
    echo "All tests passed!"
fi

EOF
RUN chmod +x /usr/local/bin/run-tests-in-docker.sh
USER appuser

# Copy source
COPY --chown=appuser:appuser . .

# Build and test in one step
CMD ["/usr/local/bin/run-tests-in-docker.sh"]
