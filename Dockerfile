FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    INDEX_DIR=/app/data/index

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/index \
    && chown -R appuser:appuser /app \
    && apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=appuser:appuser src /app/src
COPY --chown=appuser:appuser data/policy /app/data/policy
COPY --chown=appuser:appuser data/amendments /app/data/amendments
COPY --chown=appuser:appuser scripts /app/scripts
COPY --chown=appuser:appuser evaluation /app/evaluation
COPY --chown=appuser:appuser tests /app/tests
COPY --chown=appuser:appuser pytest.ini /app/pytest.ini

USER root

COPY --chmod=755 scripts/docker_entrypoint.sh /app/scripts/docker_entrypoint.sh
RUN sed -i 's/\r$//' /app/scripts/docker_entrypoint.sh

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
CMD ["idle"]
