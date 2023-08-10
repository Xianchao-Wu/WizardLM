#########################################################################
# File Name: mbpp_1.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Aug 10 06:39:18 2023
#########################################################################
#!/bin/bash

#model="/path/to/your/model"
model="/workspace/asr/WizardLM/WizardCoder/models--WizardLM--WizardCoder-15B-V1.0/snapshots/926ca1b215c4631bc5f8c3e47173381452c23e5c"
temp=0.2 # temperature, TODO reset this 温度
max_len=2048
pred_num=200
num_seqs_per_iter=2

output_path=preds/MBPP_T${temp}_N${pred_num}
mbpp_path=data/mbpp.test.jsonl # we provide this file in data/mbpp.test.zip

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# for debug NOTE
debug=0
if [[ $debug == 1 ]]
then
    gpu=1
    start_index=0
    end_index=2

    CUDA_VISIBLE_DEVICES=$gpu python -m ipdb src/mbpp_gen.py --model ${model} \
      --start_index ${start_index} \
      --end_index ${end_index} \
      --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} \
      --N ${pred_num} \
      --max_len ${max_len} \
      --output_path ${output_path} \
      --mbpp_path ${mbpp_path}

    exit 0
fi

# 500 problems, 63 per GPU if GPU=8
index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 50))
  end_index=$(((i + 1) * 50))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python src/mbpp_gen.py --model ${model} \
      --start_index ${start_index} \
      --end_index ${end_index} \
      --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} \
      --N ${pred_num} \
      --max_len ${max_len} \
      --output_path ${output_path} \
      --mbpp_path ${mbpp_path}
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
