#!/usr/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /Users/aslihanuysal/envs/nlp_sentiment_analysis/bin/activate

model=$1
n_classes=$2

if [[ "$model" == "mbert" ]]; then
  model_name="bert-base-multilingual-cased"
elif [[ "$model" == "bert-base-turkish-sentiment-cased" ]]; then
  model_name="savasy/bert-base-turkish-sentiment-cased"
elif [[ "$model" == "bert-turkish-text-classification" ]]; then
  model_name="savasy/bert-turkish-text-classification"
fi

#"BSC-TeMU/mesinesp"
# Finetune on Text Classification
#"$SCRIPT_DIR/roberta-base-ca-cased" \

python $SCRIPT_DIR/transformers_model.py \
  --model_name_or_path ${model_name} \
  --n_classes ${n_classes} \
  --dataset_name None \
  --do_train True \
  --do_eval True \
  --do_predict False \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --load_best_model_at_end \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model loss \
  --overwrite_output_dir True \
  --seed 1 \
  --cache_dir './.cache' \
  --logging_dir "$SCRIPT_DIR/${model}-2-classes-emoji-data-log" \
  --output_dir "$SCRIPT_DIR/${model}-2-classes-emoji-data-output"