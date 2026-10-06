#!/usr/bin/env bash
# Train + test all 17 configurations on both domains (34 runs), with the exact
# hyper-parameters used in the paper (also the defaults of train.py).
#
#   bash scripts/run_all.sh                 # all 17 models
#   bash scripts/run_all.sh 9 10 17         # only these model numbers
#
# Outputs: outputs/<run>_<domain>/{best_checkpoint.pt,valid_metrics.json,test_metrics.json}
# Logs:    logs_rerun/<run>_<domain>.log
# Runs whose test_metrics.json already exists are skipped, so the script can be resumed.
# Then: python scripts/collect_results.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/punctuation}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs}"
LOG_DIR="${LOG_DIR:-$ROOT/logs_rerun}"
PYTHON="${PYTHON:-python}"
DOMAINS="${DOMAINS:-Novels News}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

# number | run name | train.py flags
MODELS=(
    "1|mbert|--backbone mbert"
    "2|velectra|--backbone velectra"
    "3|vibert|--backbone vibert"
    "4|xlmr|--backbone xlmr"
    "5|mbert_bilstm|--backbone mbert --use_bilstm"
    "6|velectra_bilstm|--backbone velectra --use_bilstm"
    "7|vibert_bilstm|--backbone vibert --use_bilstm"
    "8|xlmr_bilstm|--backbone xlmr --use_bilstm"
    "9|mbert_crf|--backbone mbert --use_crf"
    "10|velectra_crf|--backbone velectra --use_crf"
    "11|vibert_crf|--backbone vibert --use_crf"
    "12|xlmr_crf|--backbone xlmr --use_crf"
    "13|mbert_bilstm_crf|--backbone mbert --use_bilstm --use_crf"
    "14|velectra_bilstm_crf|--backbone velectra --use_bilstm --use_crf"
    "15|vibert_bilstm_crf|--backbone vibert --use_bilstm --use_crf"
    "16|xlmr_bilstm_crf|--backbone xlmr --use_bilstm --use_crf"
    "17|phopunct|--backbone phobert --use_bilstm --use_crf"
)

selected=" $* "
cd "$ROOT"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r num run flags <<< "$entry"
    if [ $# -gt 0 ] && [[ "$selected" != *" $num "* ]]; then continue; fi
    for domain in $DOMAINS; do
        dom=$(echo "$domain" | tr '[:upper:]' '[:lower:]')
        out="$OUT_ROOT/${run}_${dom}"
        if [ -f "$out/test_metrics.json" ]; then
            echo "[skip] #$num $run $domain (already done)"; continue
        fi
        echo "[$(date '+%F %T')] #$num $run on $domain"
        # shellcheck disable=SC2086
        "$PYTHON" train.py $flags \
            --data_dir "$DATA_ROOT/$domain" --output_dir "$out" \
            --device cuda --fp16 2>&1 | tee "$LOG_DIR/${run}_${dom}.log"
    done
done
echo "Done. Summarise with: python scripts/collect_results.py --outputs $OUT_ROOT"
