# Template-free Articulated Gaussian Splatting for Real-time Reposable Dynamic View Synthesis

## Install

```bash 
conda env create -n enviroment.yaml
conda activate SK_GS
cd <project root>
cd extenstion/_C
mkdir build
cd build
cmake ..
make -j
```

## Data Prepare

1. Download [D-NeRF dataset](https://github.com/albertpumarola/D-NeRF). Unzip the downloaded data tor prooject root data dir in order to train.
2. Download [WIM dataset](https://github.com/NVlabs/watch-it-move) and Unzip to <data> dir.
3. Prepare ZJU Mocap dataset as [watch-it-move](https://github.com/NVlabs/watch-it-move)
4. Dataset structure

```text
<project root>
├── data
│   ├── DNeRF  
│   │   ├── mutant
│   │   ├── standup 
│   │   ├── ...
│   ├── WIM  
│   │   ├── atlas
│   │   ├── baxter 
│   │   ├── ...
│   ├── zju  
│   │   ├── 313
│   │   ├── ...
```

## Train and Test

``` bash
python train.py -c exps/d_nerf.yaml --scene hook
python test.py -c exps/d_enrf.yaml --scene hook --load results/DNeRF/last.pth
```

## GUI

```bash
python gui.py -c exps/d_enrf.yaml --scene hook --load results/DNeRF/last.pth
```