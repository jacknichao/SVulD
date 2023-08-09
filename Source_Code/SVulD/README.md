Here we provide fine-tune settings for SVulD, whose results are reported in the paper.

```shell
# Training
 python run.py \
    --output_dir saved_models/r_drop \
    --model_name_or_path microsoft/unixcoder-base-nine \
    --do_train \
    --train_data_file ./dataset/train.jsonl \
    --eval_data_file ./dataset/valid.jsonl \
    --num_train_epochs 20 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 99 \
    --r_drop
    
# Evaluating	
python run.py \
    --output_dir saved_models/r_drop \
    --model_name_or_path microsoft/unixcoder-base-nine \
    --do_test \
    --test_data_file ./dataset/test.jsonl \
    --block_size 400 \
    --eval_batch_size 16 \
    --seed 99
```
