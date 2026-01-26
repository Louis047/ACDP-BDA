#!/bin/bash
# Full ACADP Pipeline - End-to-End Execution

set -e

INPUT="${1:-data/raw_dataset.csv}"
EPSILON="${2:-1.0}"
OUTPUT_DIR="${3:-output}"

echo "=========================================="
echo "ACADP FULL PIPELINE"
echo "=========================================="
echo "Input: $INPUT"
echo "Privacy Budget (ε): $EPSILON"
echo "Output Directory: $OUTPUT_DIR"
echo ""

python src/pipeline.py \
    --input "$INPUT" \
    --epsilon "$EPSILON" \
    --output-dir "$OUTPUT_DIR" \
    --log-level INFO

echo ""
echo "Full pipeline execution complete!"
echo "Results saved to: $OUTPUT_DIR"
