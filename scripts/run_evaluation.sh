#!/bin/bash
# Workflow 4: Evaluation & Validation

set -e

ORIGINAL="${1:-data/preprocessed.parquet}"
ACADP="${2:-data/acadp_privatized.parquet}"
BASELINE="${3:-data/baseline_privatized.parquet}"
OUTPUT="${4:-results/comparison.json}"

echo "=========================================="
echo "WORKFLOW 4: EVALUATION & VALIDATION"
echo "=========================================="
echo "Original: $ORIGINAL"
echo "ACADP: $ACADP"
echo "Baseline: $BASELINE"
echo "Output: $OUTPUT"
echo ""

python -m src.evaluation.comparisons \
    --original "$ORIGINAL" \
    --acadp "$ACADP" \
    --baseline "$BASELINE" \
    --output "$OUTPUT"

echo ""
echo "Evaluation complete!"
