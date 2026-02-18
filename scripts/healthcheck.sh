#!/bin/bash
set -euo pipefail

BASE_URL="${HEALTH_CHECK_URL:-http://localhost:8000}"

# Core API health
curl -sf "${BASE_URL}/health" > /dev/null || { echo "FAIL: /health"; exit 1; }

# Models endpoint
curl -sf "${BASE_URL}/api/v1/models" > /dev/null || { echo "FAIL: /api/v1/models"; exit 1; }

# Narrator (only when enabled)
if [ "${NARRATOR_ENABLED:-false}" = "true" ]; then
  curl -sf "${BASE_URL}/api/v1/narrator/status" > /dev/null || { echo "FAIL: /api/v1/narrator/status"; exit 1; }
fi

echo "OK: all checks passed"
