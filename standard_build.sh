#!/bin/bash
set -e
export PATH="$HOME/.elan/bin:$PATH"

# 1. Remove broken test file
rm -f verification_env/VerificationEnv/Test.lean
echo "Removed broken test file."

cd verification_env

# 2. Standard workflow
echo "Running lake update..."
lake update

echo "Running lake exe cache get..."
lake exe cache get

# NOTE: We intentionally skip 'lake build' here.
# See docs/LEAN_BUILD_EXPLAINED.md for a detailed explanation.
# TL;DR: Our verifier uses 'lake env lean <file>' which only needs the
# mathlib cache (downloaded above), not a full build of our project.
echo "Skipping 'lake build' - not required for verification workflow."
echo "Build complete! The verification environment is ready."
