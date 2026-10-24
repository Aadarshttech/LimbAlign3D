#!/usr/bin/env bash
set -e

# LimbAlign3D: Data acquisition script for VSDFullBodyBoneReconstruction (Zenodo 8302449)
TARGET_DIR="${1:-data/raw}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Checking for zenodo_get..."
if ! command -v zenodo_get &> /dev/null; then
    echo "Installing zenodo_get..."
    pip install zenodo_get
fi

echo "Downloading VSDFullBodyBoneReconstruction dataset (Zenodo ID: 8302449)..."
zenodo_get 8302449

echo "Verifying MD5 checksums..."
if [ -f "md5sums.txt" ]; then
    md5sum -c md5sums.txt
fi

echo "Download complete in $TARGET_DIR"

