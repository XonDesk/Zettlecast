# The primary's API (ROLE=primary) in OrbStack: no GPU work, search's query model on the CPU.
# Built from the repository root: docker compose (see deploy/macos/README.md, "Containers").
FROM python:3.12-slim

# tzdata: queue and heartbeat times are local and compared with the Mac's (TZ is set by compose)
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# editable: the app reads /app/.env (mounted) next to src/, as from a checkout; CPU-only torch
RUN uv pip install --system --no-cache -e . \
        --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# float16: half the memory of float32 on the CPU, as fast (bfloat16 is ~80x slower on this CPU)
ENV PYTHONUNBUFFERED=1 HF_HOME=/hf WAKE_SENDER=helper MODEL_DTYPE=float16
EXPOSE 8000
CMD ["uvicorn", "zettlecast.main:app", "--host", "0.0.0.0", "--port", "8000"]
