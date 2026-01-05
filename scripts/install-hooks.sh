#!/bin/bash
#===============================================================================
# Install Git hooks for security auditing
# Run this once after cloning the repository
#===============================================================================

set -e

HOOKS_DIR=".githooks"
GIT_HOOKS_DIR=".git/hooks"

echo "Installing Git hooks..."

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository root"
    exit 1
fi

# Copy hooks
for hook in "$HOOKS_DIR"/*; do
    if [ -f "$hook" ]; then
        hook_name=$(basename "$hook")
        echo "  Installing $hook_name..."
        cp "$hook" "$GIT_HOOKS_DIR/$hook_name"
        chmod +x "$GIT_HOOKS_DIR/$hook_name"
    fi
done

echo ""
echo "✓ Git hooks installed successfully"
echo ""
echo "The following checks will run before each commit:"
echo "  - Hardcoded home directory paths"
echo "  - Local/private IP addresses (192.168.x.x, 10.x.x.x, 100.x.x.x)"
echo "  - API keys, tokens, passwords"
echo "  - Email addresses"
echo "  - SSH private keys"
echo "  - Other credential patterns"
echo ""
echo "To bypass (not recommended): git commit --no-verify"
