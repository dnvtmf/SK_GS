#!/usr/bin/env bash
scenes=(hellwarrior hook jumpingjacks mutant standup trex) # bouncingballs lego
gpus=(0 1 2 3 4 5 6 7)
args=()
num_scenes=${#scenes[@]}
num_gpus=${#gpus[@]}
echo "There are ${num_gpus} gpus and ${num_scenes} scenes"
ablation_case=num_sp

for (( i = 0;  i < ${num_gpus}; ++i ))
do
    gpu_id="gpu${gpus[$i]}"
    if ! screen -ls ${gpu_id}
    then
        echo "create ${gpu_id}"
        screen -dmS ${gpu_id}
    fi
    screen -S ${gpu_id} -p 0 -X stuff "^M"
    screen -S ${gpu_id} -p 0 -X stuff "export CUDA_VISIBLE_DEVICES=${gpus[$i]}^M"
    screen -S ${gpu_id} -p 0 -X stuff "cd ~/wan_code/NeRF^M"
done
screen -ls%

echo "The configuares in ${ablation_case}:"
ls exps/sk_gs/${ablation_case}

k=0
for (( i=0; i < num_scenes; ++i ))
do
  for exp in $(ls exps/sk_gs/${ablation_case}/)
  do
      if [[ ${exp##*.} != 'yaml' || ${exp:0} == '_' ]]
      then
        continue
      fi
      if [[ -e results/SK_GS/${ablation_case}/${scenes[i]}/${exp%%.yaml}/last.pth ]]
      then
          continue
      fi
      gpu_id=${gpus[$(( k % num_gpus ))]}
      echo "use gpu${gpu_id} on scene: ${scenes[i]} for exp: ${ablation_case}/${exp} "
      screen -S gpu${gpu_id} -p 0 -X stuff "^M"
      screen -S gpu${gpu_id} -p 0 -X stuff \
        "python3 train.py -c exps/sk_gs/${ablation_case}/${exp} --scene=${scenes[i]} ${args[*]} ^M"
      k=$((k + 1))
  done
done










