#!/bin/bash
VERSION=$(churing --version 2>/dev/null || echo "?")
cat <<EOF

  Churing $VERSION  —  a functional language (Church + Turing)
  ─────────────────────────────────────────────────────────────
  Run an example:    churing examples/01_hello.ch
  List all examples: ls examples/
  Browse a file:     cat examples/03_functions.ch

EOF
exec /bin/bash "$@"
