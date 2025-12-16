#!/bin/bash
set -e

# Source .profile or .elan/env if available
if [ -f "$HOME/.elan/env" ]; then
    source "$HOME/.elan/env"
fi

cd verification_env

echo "Step 1: Lake build (initial)..."
lake build

echo "Step 2: Get mathlib cache..."
lake exe cache get

echo "Step 3: Lake build (final)..."
lake build

echo "Environment setup complete."
