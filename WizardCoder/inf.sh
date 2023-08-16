#########################################################################
# File Name: inf.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Jul 28 07:55:20 2023
#########################################################################
#!/bin/bash

bmodel="/workspace/asr/WizardLM/WizardCoder/models--WizardLM--WizardCoder-15B-V1.0/snapshots/926ca1b215c4631bc5f8c3e47173381452c23e5c"
inpath="./data/in.jsonl"
outpath="./data/out.jsonl"

python -m ipdb src/inference_wizardcoder.py \
    --base_model $bmodel \
    --input_data_path $inpath \
    --output_data_path $outpath

