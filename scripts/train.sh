#!/usr/bin/env bash
set -e

# LimbAlign3D: Baseline training launch script
DATA_DIR="${1:-data/processed}"
CONFIG="${2:-configs/splits.json}"
OUTPUT_DIR="${3:-checkpoints}"

echo "Launching lower-limb bone segmentation training..."
echo "Data Directory: $DATA_DIR"
echo "Split Config:   $CONFIG"
echo "Output Directory: $OUTPUT_DIR"

python -m limbalign.cli train \
    --data_dir "$DATA_DIR" \
    --split "$CONFIG" \
    --epochs 100 \
    --batch_size 2 \
    --output "$OUTPUT_DIR"

