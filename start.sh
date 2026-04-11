#!/bin/bash
set -e

echo "=== OpenCure Startup ==="

# Use configurable data dir (Railway volume or local)
DATA_ROOT="${OPENCURE_DATA_DIR:-data}"
export OPENCURE_DATA_DIR="$DATA_ROOT"

# Download DRKG if not present
if [ ! -f "$DATA_ROOT/drkg/drkg.tsv" ]; then
    echo "Data not found at $DATA_ROOT/drkg/drkg.tsv — downloading..."
    mkdir -p "$DATA_ROOT/drkg"

    DRKG_URL="https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"
    curl -L "$DRKG_URL" -o "$DATA_ROOT/drkg/drkg.tar.gz"
    tar -xzf "$DATA_ROOT/drkg/drkg.tar.gz" -C "$DATA_ROOT/drkg" --strip-components=0
    rm -f "$DATA_ROOT/drkg/drkg.tar.gz"

    # Download embeddings
    EMB_URL="https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/embed.tar.gz"
    curl -L "$EMB_URL" -o "$DATA_ROOT/drkg/embed.tar.gz" || true
    if [ -f "$DATA_ROOT/drkg/embed.tar.gz" ]; then
        tar -xzf "$DATA_ROOT/drkg/embed.tar.gz" -C "$DATA_ROOT/drkg"
        rm -f "$DATA_ROOT/drkg/embed.tar.gz"
    fi

    echo "Data download complete."
else
    echo "Data found at $DATA_ROOT/drkg/drkg.tsv"
fi

echo "Starting OpenCure web server..."
exec python -m opencure.web.run
