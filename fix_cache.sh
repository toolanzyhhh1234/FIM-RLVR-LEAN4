#!/bin/bash
set -e
export PATH="$HOME/.elan/bin:$PATH"

cd verification_env
echo "Running lake exe cache get..."
lake exe cache get
echo "\nChecking if cache files exist..."
find .lake/packages/mathlib/.lake/build -name "*.olean" | head -n 5
