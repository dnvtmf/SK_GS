# Template-free Articulated Gaussian Splatting for Real-time Reposable Dynamic View Synthesis

[🌐Project Page](https://dnvtmf.github.io/SK_GS) | [🖨️ArXiv](https://arxiv.org/pdf/2610.02133) | [📰Paper](https://arxiv.org/pdf/2610.02133)

## Install

```bash 
# install requirements
conda env create -n enviroment.yaml
conda activate SK_GS
# build extension
cd <project root>
cd extenstion/_C
mkdir build
cd build
cmake ..
make -j
```

## Data Prepare

1. Download [D-NeRF dataset](https://github.com/albertpumarola/D-NeRF). Unzip the downloaded data tor prooject root data
   dir in order to train.
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

## Thanks

Thanks to the authors
of [3D Gaussians](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [Deformable-3D-Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians)
and  [SC-GS](https://github.com/yihua7/SC-GS) for their excellent code.

## ✏️ Citateion

```text
@InProceedings{SK-GS,
  title = 	 {Template-free Articulated Gaussian Splatting for Real-time Reposable Dynamic View Synthesis},
  author =       {Wan, Diwen and Wang, Yuxiang and Lu, Ruijie and Zeng, Gang},
  booktitle = 	 {NeurIPS},
  year = 	 {2024},
}
```

```text
@InProceedings{SP-GS,
  title = 	 {Superpoint Gaussian Splatting for Real-Time High-Fidelity Dynamic Scene Reconstruction},
  author =       {Wan, Diwen and Lu, Ruijie and Zeng, Gang},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {49957--49972},
  year = 	 {2024},
}
```
