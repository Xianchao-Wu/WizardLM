#########################################################################
# File Name: mbpp_3_a100.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Aug 15 07:29:53 2023
#########################################################################
#!/bin/bash

jsonfn="/workspace/asr/WizardLM/WizardCoder/preds/MBPP_T0.2_N2.jsonl"

#accelerate launch  main.py   --tasks mbpp \
python -m ipdb main.py --tasks mbpp \
	--allow_code_execution \
	--load_generations_path $jsonfn \
    --n_samples 2 \
	--model incoder-temperature-08
