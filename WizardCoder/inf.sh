#########################################################################
# File Name: inf.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Jul  3 04:13:18 2023
#########################################################################
#!/bin/bash

ckpt="/workspace/asr/Llama-X/src/checkpoints_wcode/models--WizardLM--WizardCoder-15B-V1.0/snapshots/69e87732535159460155972c3fac32a6241cc0ca"
indata="/workspace/asr/WizardLM/WizardCoder/data/in.data.jsonl"
outdata="/workspace/asr/WizardLM/WizardCoder/data/out.res.jsonl"

python -m ipdb src/inference_wizardcoder.py \
    --base_model $ckpt \
    --input_data_path $indata \
    --output_data_path $outdata 
