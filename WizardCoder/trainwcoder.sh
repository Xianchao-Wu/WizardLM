#########################################################################
# File Name: trainwcoder.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Aug 18 08:03:48 2023
#########################################################################
#!/bin/bash

#!/bin/bash

data="/workspace/asr/WizardLM/WizardCoder/data/code_alpaca_20k.json"
#outdir="/workspace/asr/Llama-X/src/checkpoints_wcode"
outdir="/workspace/asr/WizardLM/WizardCoder/ckpts"

#deepspeed src/train_wizardcoder.py \
python -m ipdb src/train_wizardcoder.py \
    --model_name_or_path "bigcode/starcoder" \
    --data_path $data \
    --output_dir $outdir  \
	--cache_dir "/workspace/asr/WizardLM/WizardCoder" \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 30 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True

