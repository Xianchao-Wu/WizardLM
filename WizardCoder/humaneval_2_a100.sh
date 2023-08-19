#########################################################################
# File Name: humaneval_2_a100.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Aug 17 23:05:32 2023
#########################################################################
#!/bin/bash

temp=0.2
pred_num=1

output_path=preds/humaneval_T${temp}_N${pred_num}

echo 'Output path: '$output_path
python src/process_humaneval.py --path ${output_path} \
	--out_path ${output_path}.jsonl \
	--add_prompt

evaluate_functional_correctness ${output_path}.jsonl
