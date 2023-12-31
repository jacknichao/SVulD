

  ```
# Training
cd linevul
python linevul_main.py \
    --output_dir=./saved_models/linevul \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --epochs 20 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 99
    
# Evaluating
cd linevul
python linevul_main.py \
    --output_dir=./saved_models/linevul \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --test_data_file=./dataset/test.jsonl \
    --block_size 400 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 99
  ```
