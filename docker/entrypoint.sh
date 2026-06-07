#!/bin/bash
VERSION=$(churing --version 2>/dev/null || echo "?")
cat <<EOF

  Churing $VERSION  —  an AI-native functional language (Church + Turing)
  ─────────────────────────────────────────────────────────────
  Run a file:  churing path/to/file.ch
  Docs:        https://github.com/petrucristescu/churing

EOF
exec /bin/bash "$@"
