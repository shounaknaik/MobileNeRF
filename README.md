# Real-Time Neural Light Field on Mobile Devices


This work is an extension of the following work named `MobileR2L`
### [Project](https://snap-research.github.io/MobileR2L/) | [ArXiv](https://arxiv.org/abs/2212.08057) | [PDF](https://arxiv.org/pdf/2212.08057.pdf) 





# Abstract

We have setup the complex pipeline of NGP_PL and MobileR2L on our systems. Furthermore we have also setup an iterative pruning pipeline to prune the NeRF model. 


# Dataset
Download the example data: `lego`
```
sh script/download_example_data.sh
```

# Knowledge Distillation

For this part, we require `TinyCudaNN` which requires a running GPU instance and thus we setup and ran this part on a AWS instance. Refer the original `MobileR2L` github for details on how to set this up.

This part creates pseudo images from the teacher model which the student then uses to do imitation learning.

To do this, you need to first setup  `NGP_PL`
Then you need to download the dataset.

Create pseudo images by going into the `ngp_pl` folder. 
Download the teacher checkpoint from the `'ngp_pl` [github](https://github.com/kwea123/ngp_pl).


Run the following, this allows us to create 5k images from the trained teacher model,it takes in the weights for the trained teacher model and you can specify the number of GPUs
```
export ROOT_DIR=dataset/nerf_synthetic/
python3 train.py \
    --root_dir $ROOT_DIR/lego \
    --exp_name Lego_Pseudo  \
    --save_pseudo_data \
    --n_pseudo_data 5000 --weight_path ckpts/nerf/lego/epoch=29_slim.ckpt \
    --save_pseudo_path Pseudo/lego --num_gpu 1
```

# Training the Student

To do this we switch to `MobileR2L` and run the following

```
nGPU=1
scene=lego

python3 -m torch.distributed.launch --nproc_per_node=$nGPU --use_env main.py \
    --project_name $scene \
    --dataset_type Blender \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene  \
    --root_dir dataset/nerf_synthetic \
    --run_train \
    --num_workers 12 \
    --batch_size 10 \
    --num_iters 300000 \
    --input_height 100 \
    --input_width 100 \
    --output_height 800 \
    --output_width 800 \
    --scene $scene \
    --i_testset 10000 \
    --amp \
    --lrate 0.0005 \
    --i_weights 100
```

This allows us to train the student on the pseudo generated data, we can set the `batch_size`, `num_iteration`,` lrate` etc. The parameters `i_weights` and `i_testset` are critical for saving the model checkpoint after the set amount of iterations.

# Iterative Pruning.

Here we prune the model globally by using L! Policy.
The code for this is written in  `prune.py`. 
This code expects a `ckpt.tar` from the trained student model,
It prunes the entire model by a set amount and we can change this amount.
It also makes the pruning permanent and saved the pruned model as `ckpt_pruned.tar`

Go through the simple code of pruning and run `python3 pruning.py` accordinlgy.

Once we get the pruned model, we want to finetune it and thus we again resort to training the student model script.

Run 
```
nGPU=1
scene=lego

python3 -m torch.distributed.launch --nproc_per_node=$nGPU --use_env main.py \
    --project_name $scene \
    --dataset_type Blender \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene  \
    --root_dir dataset/nerf_synthetic \
    --run_train \
    --num_workers 12 \
    --batch_size 10 \
    --ckpt_dir ckpt_pruned.tar \
    --num_iters 50000 \
    --input_height 100 \
    --input_width 100 \
    --output_height 800 \
    --output_width 800 \
    --scene $scene \
    --i_testset 10000 \
    --amp \
    --lrate 5e-6 \
    --i_weights 100
```

Here the notable changes are the addition of the `ckpt_dir` flag. This allows us to give a pretrained network for finetuning. We reduce the `num_iters` to `50000` and also decrease the learning rate to `5e-6`. This is standard for finetuning purposes.

We did this 3 times in iteration and have reported our result in the report.


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
