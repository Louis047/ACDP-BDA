#!/bin/bash
# Workflow 1: Data Ingestion & Preprocessing

set -e

INPUT="${1:-data/raw_dataset.csv}"
OUTPUT="${2:-data/preprocessed.parquet}"

echo "=========================================="
echo "WORKFLOW 1: DATA INGESTION & PREPROCESSING"
echo "=========================================="
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo ""

python -m src.ingestion.load_data \
    --input "$INPUT" \
    --output "$OUTPUT"

echo ""
echo "Preprocessing complete!"
