#!/usr/bin/env bash
set -euo pipefail

uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked mypy
uv run --locked pytest
