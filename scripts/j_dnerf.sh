#!/usr/bin/env bash
scenes=(hellwarrior  hook  jumpingjacks mutant  standup  trex) # bouncingballs lego
gpus=(0 1 3 4 5 6 7 8 9)
args=()
test_args=()
cfg_path=sk_gs/d_nerf3.yaml
out_dir=SK_GS/DNeRF3_RBF
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
#    if [[ -e  "results/${out_dir}/${scenes[i]}/checkpoint.pth" ]]
#    then
#        screen -S gpu${gpu_id} -p 0 -X stuff \
#            "python3 gaussian_train.py -c exps/${cfg_path} --scene=${scenes[i]} ${args[*]} \
#            --resume results/${out_dir}/${scenes[i]}/checkpoint.pth ^M"
#    else
#        echo "skip"
        screen -S gpu${gpu_id} -p 0 -X stuff \
        "python3 gaussian_train.py -c exps/${cfg_path} --scene=${scenes[i]} ${args[*]} ^M"
#    fi
    screen -S gpu${gpu_id} -p 0 -X stuff \
        "python3 gs_test.py -c exps/${cfg_path} \
        --load results/${out_dir}/${scenes[i]}/last.pth \
        --scene ${scenes[i]} --load-no-strict ${test_args[*]} ^M"
done
