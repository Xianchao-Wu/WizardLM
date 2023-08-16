#########################################################################
# File Name: humaneval_1_dgx1.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Aug 16 08:45:55 2023
#########################################################################
#!/bin/bash

#model="/path/to/your/model"
model="/workspace/asr/WizardLM/WizardCoder/models--WizardLM--WizardCoder-15B-V1.0/snapshots/926ca1b215c4631bc5f8c3e47173381452c23e5c"
temp=0.2
max_len=2048
pred_num=1 #200
num_seqs_per_iter=1 #2

output_path=preds/humaneval_T${temp}_N${pred_num}

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems, 21 per GPU if GPU=8
index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 21))
  end_index=$(((i + 1) * 21))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python src/humaneval_gen.py --model ${model} \
      --start_index ${start_index} \
      --end_index ${end_index} \
      --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} \
      --N ${pred_num} \
      --max_len ${max_len} \
      --output_path ${output_path}
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
