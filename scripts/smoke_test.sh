#!/usr/bin/env bash
# CPU sanity check (~minutes): train 1 epoch on 50 chunks for one model of each
# family (softmax / CRF / PhoPunct), then run evaluate.py and predict.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
SMOKE="${SMOKE:-$ROOT/smoke}"
DOMAIN="${DOMAIN:-News}"

"$PYTHON" scripts/make_smoke_subset.py --src_dir "data/punctuation/$DOMAIN" --dst_dir "$SMOKE/data/$DOMAIN"
COMMON="--data_dir $SMOKE/data/$DOMAIN --device cpu --max_seq_length 128 --train_batch_size 4
        --eval_batch_size 8 --num_train_epochs 1 --logging_steps 5
        --max_train_examples 50 --max_eval_examples 20"

# shellcheck disable=SC2086
"$PYTHON" train.py --backbone xlmr $COMMON --output_dir "$SMOKE/out/xlmr"
# shellcheck disable=SC2086
"$PYTHON" train.py --backbone vibert --use_bilstm --use_crf $COMMON --output_dir "$SMOKE/out/vibert_bilstm_crf"
# PhoBERT-base keeps the PhoPunct smoke test light on CPU.
# shellcheck disable=SC2086
"$PYTHON" train.py --backbone phobert --model_name_or_path vinai/phobert-base --use_bilstm --use_crf \
    $COMMON --output_dir "$SMOKE/out/phopunct"

"$PYTHON" evaluate.py --checkpoint "$SMOKE/out/phopunct/best_checkpoint.pt" \
    --data_dir "$SMOKE/data/$DOMAIN" --max_seq_length 128 --max_eval_examples 20 --device cpu
"$PYTHON" predict.py --checkpoint "$SMOKE/out/phopunct/best_checkpoint.pt" --device cpu \
    --text "bạn có khỏe không tôi rất nhớ bạn"
echo "Smoke test OK."
