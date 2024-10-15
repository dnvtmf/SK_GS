#!/usr/bin/env bash
set -e
ROOT=$(
  cd $(dirname $0)/..
  pwd
)
scenes=(hellwarrior hook jumpingjacks mutant standup trex) # bouncingballs lego
gpus=(0 1 2 3 4 5 6 7 8)
args=()
num_scenes=${#scenes[@]}
num_gpus=${#gpus[@]}
echo "There are ${num_gpus} gpus and ${num_scenes} scenes"
#ablation_case=num_sp
#ablation_case=warp
#ablation_case=sp_control
#ablation_case=num_knn
#ablation_case=lr_deform
#ablation_case=sk_knn_num
#ablation_case=loss_sparse
#ablation_case=loss_smooth
#ablation_case=loss_joint
#ablation_case=loss_cmp_p
ablation_case=loss_cmp_t

for ((i = 0; i < ${num_gpus}; ++i)); do
  gpu_id="gpu${gpus[$i]}"
  if ! screen -ls ${gpu_id}; then
    echo "create ${gpu_id}"
    screen -A -dmS ${gpu_id}
  fi
  screen -S ${gpu_id} -p 0 -X stuff "^M"
  screen -S ${gpu_id} -p 0 -X stuff "conda activate SK_GS^M"
  screen -S ${gpu_id} -p 0 -X stuff "export CUDA_VISIBLE_DEVICES=${gpus[$i]}^M"
  screen -S ${gpu_id} -p 0 -X stuff "cd ${ROOT}^M"
done
screen -ls%

echo "The configuares in ${ablation_case}:"
ls exps/${ablation_case}

#find ${ROOT}/results -name '*init*.pth' | xargs rm
#find ${ROOT}/results -name '*checkpoint*.pth' | xargs rm
#find ${ROOT}/results -name 'best.pth' | xargs rm
#find ${ROOT}/results -name 'last.pth' | xargs rm
df -h ${ROOT}

k=0
for ((i = 0; i < num_scenes; ++i)); do
  for exp in $(ls exps/${ablation_case}/); do
    if [[ ${exp##*.} != 'yaml' || ${exp:0:1} == '_' ]]; then
      continue
    fi
    gpu_id=${gpus[$((k % num_gpus))]}
    echo "use gpu${gpu_id} on scene: ${scenes[i]} for exp: ${ablation_case}/${exp} "
    screen -S gpu${gpu_id} -p 0 -X stuff "^M"
    if [[ ! -e results/${ablation_case}/${scenes[i]}/${exp%%.yaml}/last.pth ]]; then
      screen -S gpu${gpu_id} -p 0 -X stuff \
        "python3 train.py -c exps/${ablation_case}/${exp} --scene=${scenes[i]} ${args[*]} ^M"
    fi
    if [[ ! -e results/${ablation_case}/${scenes[i]}/${exp%%.yaml}/results.json ]]; then
      screen -S gpu${gpu_id} -p 0 -X stuff \
        "python3 test.py -c exps/${ablation_case}/${exp} \
          --load results/${ablation_case}/${scenes[i]}/${exp%%.yaml}/last.pth \
          --scene ${scenes[i]} --load-no-strict ${test_args[*]} ^M"
    fi
    k=$((k + 1))
  done
done
