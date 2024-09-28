#!/usr/bin/env bash
scenes=(atlas baxter spot cassie iiwa nao pandas)
gpus=(0 1 2 3 4 5 6 7)
args=()
test_args=()
num_scenes=${#scenes[@]}
num_gpus=${#gpus[@]}
echo "There are ${num_gpus} gpus and ${num_scenes} scenes"

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

for (( i=0; i < num_scenes; ++i ))
do
    gpu_id=${gpus[$(( i % num_gpus ))]}
    echo "use gpu${gpu_id} on scene: ${scenes[i]} "
    screen -S gpu${gpu_id} -p 0 -X stuff "^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 skeleton_train.py -c exps/sk_gs/wim_512.yaml --scene=${scenes[i]} ${args[*]} ^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 sp_gs_test.py -c exps/sk_gs/wim_512.yaml \
        --load results/SK_GS/WIM512/${scenes[i]}/last.pth \
        --scene ${scenes[i]} --load-no-strict ${test_args[*]} ^M"
done
