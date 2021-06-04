# Training and Testing

[English](TrainTest.md)

Please run the commands in the root path of `FS-IL`. <br>
In general, both the training and testing include the following steps:

1. Prepare datasets. 
1. Modify config files. The config files are under the `options` folder. For more specific configuration information, please refer to [Config](Config.md)
1. Run commands. Use [Training Commands](#Training-Commands) or [Testing Commands](#Testing-Commands) accordingly.

#### 目录

1. [Training Commands (Training the base model)](#Training-Commands)
    1. [Single GPU Training](#Single-GPU-Training)
1. [Incremental Testing Commands](#Testing-Commands)
    1. [Single GPU Testing](#Single-GPU-Testing)

## Training Commands

### Single GPU Training

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python train.py -opt ./options/train/cifar/bases60_noBuffer/bases60_CFRPModel_baseline_SGD.yml

## Incremental Testing Commands

### Single GPU Testing

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python incremental_procedure.py -opt ./options/incremental/cifar/bases60_noBuffer/incremental_res18_cifar_baseline_5shots_SGD.yml