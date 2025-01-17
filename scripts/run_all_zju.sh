#!/usr/bin/env bash
set -e
ROOT=$(
  cd $(dirname $0)/..
  pwd
)
scenes=(366 377 381 384 387)
gpus=(0 1 3 4 6 7)
args=()
test_args=()
num_scenes=${#scenes[@]}
num_gpus=${#gpus[@]}
echo "There are ${num_gpus} gpus and ${num_scenes} scenes"

for ((i = 0; i < ${num_gpus}; ++i)); do
  gpu_id="gpu${gpus[$i]}"
  if ! screen -ls ${gpu_id}; then
    echo "create ${gpu_id}"
    screen -dmS ${gpu_id}
  fi
  screen -S ${gpu_id} -p 0 -X stuff "^M"
  screen -S ${gpu_id} -p 0 -X stuff "conda activate SK_GS^M"
  screen -S ${gpu_id} -p 0 -X stuff "export CUDA_VISIBLE_DEVICES=${gpus[$i]}^M"
  screen -S ${gpu_id} -p 0 -X stuff "cd ${ROOT}^M"
done
screen -ls%

for ((i = 0; i < num_scenes; ++i)); do
  gpu_id=${gpus[$((i % num_gpus))]}
  echo "use gpu${gpu_id} on scene: ${scenes[i]} "
  screen -S gpu${gpu_id} -p 0 -X stuff "^M"
  screen -S gpu${gpu_id} -p 0 -X stuff \
    "python3 train.py -c exps/zju.yaml --scene=${scenes[i]} ${args[*]} ^M"
  screen -S gpu${gpu_id} -p 0 -X stuff \
    "python3 test.py -c exps/zju.yaml \
        --load results/ZJU/${scenes[i]}/best.pth \
        --scene ${scenes[i]} --load-no-strict ${test_args[*]} ^M"
done
