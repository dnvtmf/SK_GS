#!/usr/bin/env bash
scenes=(366 377 381 384 387)
gpus=(5 6 7)
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
      "python3 gaussian_train.py -c exps/sk_gs/zju.yaml --scene=${scenes[i]} ${args[*]} ^M"
#    screen -S gpu${gpu_id} -p 0 -X stuff \
#      "python3 gaussian_train.py -c exps/sk_gs/zju_stage2.yaml --scene=${scenes[i]} ${args[*]} \
#      --load results/SK_GS/ZJU/${scenes[i]}/stage1/last.pth
#      ^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 gs_test.py -c exps/sk_gs/zju.yaml \
        --load results/SK_GS/ZJU/${scenes[i]}/last.pth \
        --scene ${scenes[i]} --load-no-strict ${test_args[*]} ^M"
done
