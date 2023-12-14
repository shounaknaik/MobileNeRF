# Real-Time Neural Light Field on Mobile Devices


This work is an extension of the following work
### [Project](https://snap-research.github.io/MobileR2L/) | [ArXiv](https://arxiv.org/abs/2212.08057) | [PDF](https://arxiv.org/pdf/2212.08057.pdf) 

<!-- <div align="center">
    <a><img src="figs/snap.svg"  height="120px" ></a>
   
</div>

This repository is for the real-time neural rendering introduced in the following CVPR'23 paper:
> **[Real-Time Neural Light Field on Mobile Devices](https://snap-research.github.io/MobileR2L/)** \
> Junli Cao <sup>1</sup>, [Huan Wang](http://huanwang.tech/) <sup>2</sup>, Pavlo Chemerys<sup>1</sup>, Vladislav Shakhrai<sup>1</sup>, Ju Hu<sup>1</sup>,  [Yun Fu](http://www1.ece.neu.edu/~yunfu/) <sup>2</sup>, Denys Makoviichuk<sup>1</sup>,  [Sergey Tulyakov](http://www.stulyakov.com/) <sup>1</sup>, [Jian Ren](https://alanspike.github.io/) <sup>1</sup> 
>
> <sup>1</sup> Snap Inc. <sup>2</sup> Northeastern University  -->



<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
We have setup the complex pipeline of NGP_PL and MobileR2L on our systems. Furthermore we have also setup an iterative pruning pipeline to prune the NeRF model. 

</details>

# Knowledge Distillation


<!-- <div align="center">
<img src="figs/Lego-Tracking.gif" width="200" height="400" />
<img src="figs/blue-.gif" width="200" height="400" />
<img src="figs/shoe_1.gif" width="200" height="400" />
</div> -->

# Update
- [x] 09/13/2023: we released the tutorial of building your own lens by utilizing [SnapML](https://docs.snap.com/lens-studio/references/guides/lens-features/machine-learning/ml-overview) and [Lens Studio](https://ar.snap.com/lens-studio-dl?utm_source=GoogleSEM&utm_medium=PAIDPLATFORM&utm_campaign=LensStudio_Brand_P0&utm_term=AR_ProductiveInnovators_CareerArtists&utm_content=LS_ProductPage&gclid=CjwKCAjwu4WoBhBkEiwAojNdXmCEiYGlvPgN1YQGTyUaCReBgasW66baB418jGOzlyql1W3eprR7fhoChuwQAvD_BwE). Check it out [here](https://github.com/Snapchat/snapml-templates/tree/main/Neural%20Rendering)! 

# Overview
This repo contains the codebases for both the teacher and student models. We use the public repo [ngp_pl](https://github.com/kwea123/ngp_pl) as the teacher for more efficient pseudo data distillation(instead of NeRF and MipNeRF as discussed in the paper).

Observed differences between `ngp` and `NeRF` teacher:
1. the training with `ngp_pl` should be less than 15 mins with 4 GPUs and pseudo data distillation for 10k images is less than 2 hours with single GPU. 
2. `ngp` renders high quality synthetic scenes than `NeRF`
3. no space contraction techniques were employed in `ngp`, thus having a inferior performance on real-world scenes

# Installation
`conda` virtual environment is recommended. The experiments were conducted on 4 Nvidia V100 GPUs. Training on one GPU should work but takes longer to converge.
## MobileR2L

```
git clone https://github.com/snap-research/MobileR2L.git

cd MobileR2L

conda create -n r2l python==3.9
conda activate r2l
conda install pip

pip install torch torchvision torchaudio
pip install -r requirements.txt 

conda deactivate
```

## NGP_PL
```
cd model/teacher/ngp_pl

# create the conda env
conda create -n ngp_pl python==3.9
conda activate ngp_pl
conda install pip

# install torch with cuda 116
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# install tiny-cuda-nn
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install torch scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+${cu116}.html

# ---install apex---
git clone https://github.com/NVIDIA/apex
cd apex
# denpency for apex
pip install packaging

## if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
## otherwise
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# ---end installing apex---


cd ../
# install other requirements
pip install -r requirements.txt

# build
pip install models/csrc/

# go to root
cd ../../../
```

# Dataset
Download the example data: `lego` and `fern`
```
sh script/download_example_data.sh
```

# Training the Teacher

```
cd model/teacher/ngp_pl

export ROOT_DIR=../../../dataset/nerf_synthetic/
python3 train.py \
     --root_dir $ROOT_DIR/lego \
     --exp_name lego\
     --num_epochs 30 --batch_size 16384 --lr 2e-2 --eval_lpips --num_gpu 4 
```
or running the bash script
```
sh benchmarking/benchmark_synthetic_nerf.sh lego
```

Once we have the teacher trained(checkpoints saved already), we can start to generate the pseudo data for MobileR2L. Depending your disk storage, the number of pseudo images could range from 2,000 to 10,000(performance varies!). Here, we set the number to 5000.

```
export ROOT_DIR=../../../dataset/nerf_synthetic/
python3 train.py \
    --root_dir $ROOT_DIR/lego \
    --exp_name Lego_Pseudo  \
    --save_pseudo_data \
    --n_pseudo_data 5000 --weight_path ckpts/nerf/lego/epoch=29_slim.ckpt \
    --save_pseudo_path Pseudo/lego --num_gpu 1
```
or running the bash script

```
sh benchmarking/distill_nerf.sh lego
```

# Training MobileR2L

```
# go to the MobileR2L directory
cd ../../../MobileR2L

conda activate r2l

# use 4 gpus for training: NeRF
sh script/benchmarking_nerf.sh 4 lego

# use 4 gpus for training: LLFF
sh script/benchmarking_llff.sh 4 orchids

```
The model will be running a day or two depending on you GPUs. When the model converges, it will automatically export the onnx files to the `Experiment/Lego_**` folder. There should be three onnx files: `Sampler.onnx`, `Embedder.onnx` and `*_SnapGELU.onnx`.

Alternatively, you can export the onnx manully by running the flowing script with `--ckpt_dir` replaced by the trained model:

```
sh script/export_onnx_nerf.sh lego path/to/ckpt

```


# Reference

Huge Thank you to 
```BibTeX
@inproceedings{cao2023real,
  title={Real-Time Neural Light Field on Mobile Devices},
  author={Cao, Junli and Wang, Huan and Chemerys, Pavlo and Shakhrai, Vladislav and Hu, Ju and Fu, Yun and Makoviichuk, Denys and Tulyakov, Sergey and Ren, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8328--8337},
  year={2023}
}
```
