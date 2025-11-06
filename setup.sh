#!/bin/sh 

: << EOF
Setup unsupervised elicitation reproduction.
Handles package management and syncing.
EOF

if ! command -v uv; then
    echo "Warning: uv package management is not installed"
    echo "To learn more about uv, see https://docs.astral.sh/uv/"
    echo "To install uv, run the following commands:"
    echo ""
    echo "  Linux/macOS:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "  macOS (homebrew):"
    echo "    brew install uv"
    echo ""
    echo "  Alternative (pip):"
    echo "    pip install uv"
    echo ""
else
    echo "Found uv package management. sycning packages..."
    uv sync
    echo "Successfully downloaded packages"
fi


