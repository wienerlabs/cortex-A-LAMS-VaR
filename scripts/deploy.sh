#!/usr/bin/env bash
# Cortex A-LAMS-VaR Deployment Script
# Usage: ./scripts/deploy.sh [staging|production]
set -euo pipefail

ENVIRONMENT="${1:-staging}"
IMAGE_NAME="cortex-risk-engine"
REGISTRY="${DOCKER_REGISTRY:-ghcr.io/cortex-agent}"

echo "=== Cortex Deploy: ${ENVIRONMENT} ==="

# 1. Build Docker image
echo "[1/5] Building Docker image..."
docker build -t "${IMAGE_NAME}:latest" .
docker tag "${IMAGE_NAME}:latest" "${REGISTRY}/${IMAGE_NAME}:${ENVIRONMENT}"
docker tag "${IMAGE_NAME}:latest" "${REGISTRY}/${IMAGE_NAME}:$(git rev-parse --short HEAD)"

# 2. Push to registry
echo "[2/5] Pushing to registry..."
docker push "${REGISTRY}/${IMAGE_NAME}:${ENVIRONMENT}"
docker push "${REGISTRY}/${IMAGE_NAME}:$(git rev-parse --short HEAD)"

# 3. Run database migrations (if applicable)
echo "[3/5] Running migrations..."
echo "  (No SQL migrations — Redis state is schema-free)"

# 4. Deploy
echo "[4/5] Deploying to ${ENVIRONMENT}..."
if [ "${ENVIRONMENT}" = "production" ]; then
    echo "  WARNING: Production deployment"
    echo "  Ensure EXECUTION_ENABLED, SIMULATION_MODE, and TRADING_MODE are correct"
fi

# Docker Compose deploy (adjust for your orchestrator: k8s, fly.io, etc.)
if [ -f "docker-compose.${ENVIRONMENT}.yml" ]; then
    docker compose -f "docker-compose.${ENVIRONMENT}.yml" up -d
elif [ -f "docker-compose.yml" ]; then
    docker compose up -d
else
    echo "  No compose file found. Push image to registry for manual deployment."
fi

# 5. Health check
echo "[5/5] Health check..."
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
for i in 1 2 3 4 5; do
    if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "  Health check passed!"
        curl -s "${HEALTH_URL}" | python3 -m json.tool 2>/dev/null || true
        echo ""
        echo "=== Deploy complete: ${ENVIRONMENT} ==="
        exit 0
    fi
    echo "  Attempt ${i}/5 — waiting..."
    sleep 5
done

echo "  Health check FAILED after 5 attempts"
exit 1
