#!/bin/sh
# Idempotent Ollama model ensure. Intended for Compose init containers.
# Usage: ollama_ensure_model.sh <model> <embedding|generation>
set -e
MODEL=$1
KIND=$2

if [ -z "$MODEL" ] || [ -z "$KIND" ]; then
    echo "Usage: ollama_ensure_model.sh <model> <embedding|generation>" >&2
    exit 2
fi

if [ "$KIND" = "embedding" ]; then
    echo "First startup downloads local AI models and builds the search index. This may take several minutes. Do not interrupt the process."
    echo "Checking embedding model ${MODEL}..."
else
    echo "Checking generation model ${MODEL}..."
fi

if ollama list 2>/dev/null | grep -F "$MODEL" >/dev/null; then
    echo "${KIND} model ${MODEL} is already present. Skipping download."
    exit 0
fi

echo "Downloading ${KIND} model ${MODEL}. This may take several minutes. Do not interrupt the process."
ollama pull "$MODEL"
echo "${KIND} model ${MODEL} download complete."
