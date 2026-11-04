#!/bin/bash
# Run the test suite in a container from the API image (no local venv on the Mac):
#   ./deploy/macos/test.sh [pytest args]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
docker image inspect zettlecast-test >/dev/null 2>&1 || \
    printf 'FROM zettlecast-api\nRUN uv pip install --system --no-cache pytest pytest-asyncio\n' | docker build -q -t zettlecast-test - >/dev/null
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$REPO:/app" -w /app zettlecast-test python -m pytest -p no:cacheprovider "${@:-tests}"
