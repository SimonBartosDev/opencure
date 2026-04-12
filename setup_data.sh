#!/bin/bash
set -e

DATA_DIR="data/drkg"
DRKG_URL="https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"

echo "=== OpenCure Data Setup ==="

# Create data directory
mkdir -p "$DATA_DIR"

# Download DRKG if not already present
if [ ! -f "$DATA_DIR/drkg.tsv" ]; then
    echo "Downloading DRKG (Drug Repurposing Knowledge Graph)..."
    curl -L "$DRKG_URL" -o "$DATA_DIR/drkg.tar.gz"
    echo "Extracting..."
    tar -xzf "$DATA_DIR/drkg.tar.gz" -C "$DATA_DIR" --strip-components=0
    rm -f "$DATA_DIR/drkg.tar.gz"
    echo "DRKG downloaded and extracted to $DATA_DIR/"
else
    echo "DRKG already present at $DATA_DIR/drkg.tsv"
fi

# Verify files
echo ""
echo "Verifying data files..."
for f in drkg.tsv; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  [OK] $DATA_DIR/$f"
    else
        echo "  [MISSING] $DATA_DIR/$f"
    fi
done

# Check for embeddings (may be in embed/ subdirectory)
EMB_DIR="$DATA_DIR/embed"
if [ -d "$EMB_DIR" ]; then
    echo "  [OK] Pretrained embeddings found at $EMB_DIR/"
else
    echo "  [INFO] No pretrained embeddings directory. Checking for separate download..."
    # DRKG embeddings might need to be downloaded separately
    EMB_URL="https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/embed.tar.gz"
    echo "  Downloading pretrained TransE embeddings..."
    curl -L "$EMB_URL" -o "$DATA_DIR/embed.tar.gz" 2>/dev/null || true
    if [ -f "$DATA_DIR/embed.tar.gz" ]; then
        tar -xzf "$DATA_DIR/embed.tar.gz" -C "$DATA_DIR"
        rm -f "$DATA_DIR/embed.tar.gz"
        echo "  [OK] Embeddings extracted to $EMB_DIR/"
    else
        echo "  [WARN] Could not download embeddings. Will need to train or find alternative."
    fi
fi

echo ""
echo "=== Setup complete ==="
echo "Next: pip install -r requirements.txt"
echo "Then: python -m opencure.cli \"Alzheimer's disease\""
