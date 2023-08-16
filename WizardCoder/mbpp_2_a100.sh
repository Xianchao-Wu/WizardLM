#########################################################################
# File Name: mbpp_2_a100.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Aug 15 06:47:10 2023
#########################################################################
#!/bin/bash

temp=0.2
pred_num=2

output_path=preds/MBPP_T${temp}_N${pred_num}
echo ${output_path}

mbpp_path=data/mbpp.test.jsonl # we provide this file in data/mbpp.test.zip

echo 'Output path: '$output_path

python -m ipdb src/process_mbpp.py --path ${output_path} \
	--out_path ${output_path}.jsonl \
	--mbpp_path ${mbpp_path} \
	--add_prompt
