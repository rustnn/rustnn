#!/bin/bash
# Install git hooks for the project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing git hooks..."

# Install pre-commit hook
cp "$SCRIPT_DIR/git-hooks/pre-commit" "$PROJECT_ROOT/.git/hooks/pre-commit"
chmod +x "$PROJECT_ROOT/.git/hooks/pre-commit"

echo "[OK] Git hooks installed."
echo ""
echo "The following hooks are now active:"
echo "  - pre-commit: runs 'cargo fmt --check' and 'cargo clippy' when Rust files are staged"
echo ""
echo "To bypass the hook (not recommended), use: git commit --no-verify"
