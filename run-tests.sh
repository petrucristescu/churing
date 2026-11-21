#!/bin/bash
# Run tests locally using Docker (no OCaml needed on host)
# Usage:
#   ./run-tests.sh              # run all tests
#   ./run-tests.sh 12_lambda    # run a single test by name

TEST_NAME="$1"

# MSYS_NO_PATHCONV prevents Git Bash on Windows from mangling /app paths
if [ -z "$TEST_NAME" ]; then
  MSYS_NO_PATHCONV=1 docker run --rm --user root -e OPAMROOTISOK=1 -v "$(pwd):/app" churing-test
else
  MSYS_NO_PATHCONV=1 docker run --rm --user root -e OPAMROOTISOK=1 -v "$(pwd):/app" churing-test bash -c "
    test_name=\"\$1\" &&
    eval \$(opam env) && dune build src/churing.exe && mkdir -p test_results &&
    test_file=\"src/test/\${test_name}.ch\" &&
    expected=\"src/test/\${test_name}.ch.out\" &&
    output_file=\"test_results/\${test_name}.out\" &&
    if [ ! -f \"\$test_file\" ]; then echo \"Test not found: \$test_file\"; exit 1; fi &&
    test_env=\"\" &&
    if [ -f \"src/test/\${test_name}.ch.env\" ]; then test_env=\$(grep -v '^ *#' \"src/test/\${test_name}.ch.env\" | tr -d '\\r' | tr '\\n' ' '); fi &&
    env \$test_env _build/default/src/churing.exe \"\$test_file\" > \"\$output_file\" 2>&1
    exit_code=\$?
    echo '--- Output ---' &&
    cat \"\$output_file\" &&
    echo '--- Result ---' &&
    if [ -f \"\$expected\" ]; then
      if diff -u \"\$output_file\" \"\$expected\"; then
        echo \"PASS: \$test_name\"
      else
        echo \"FAIL: \$test_name\"; exit 1
      fi
    elif [ \"\$exit_code\" -ne 0 ]; then
      echo \"FAIL: \$test_name (exit code \$exit_code)\"; exit 1
    else
      echo \"PASS: \$test_name\"
    fi
  " _ "$TEST_NAME"
fi
