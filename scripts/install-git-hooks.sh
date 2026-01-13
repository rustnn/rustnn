#!/bin/bash
# Install git hooks for the project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing git hooks..."

# Install pre-commit hook
cp "$SCRIPT_DIR/git-hooks/pre-commit" "$PROJECT_ROOT/.git/hooks/pre-commit"
chmod +x "$PROJECT_ROOT/.git/hooks/pre-commit"

echo "✅ Git hooks installed successfully!"
echo ""
echo "The following hooks are now active:"
echo "  - pre-commit: Runs 'cargo fmt --check' + 'cargo clippy' for Rust changes, and 'make python-ty-check' for Python changes"
echo ""
echo "To bypass the hook (not recommended), use: git commit --no-verify"
