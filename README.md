# FS-IL

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

1. Clone repo

    ```bash
    git clone https://github.com/moukamisama/FS-IL.git
    ```

1. Install dependent packages

    ```bash
    cd FS-IL
    pip install -r requirements.txt
    ```

1. Install [wandb](https://docs.wandb.com/quickstart) (optional)

Note that FS-IL is only tested in Ubuntu, and may be not suitable for Windows. You may try [Windows WSL with CUDA supports](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) :-) (It is now only available for insider build with Fast ring).

## :computer: Train and Incremental Procdure

- **Training and incremental testing commands**: Please see **[TrainTest.md](docs/TrainTest.md)** for the basic usage.

## Datasets
We evaluate our system in several datasets, including ```CUB-200-2011, CIFAR100, miniImageNet```.
Please download [CUB-200-2011](), [CIFAR100]() and [miniImageNet]().(Note: some datasets do not split the train set and test set in the original folder, the splited datasets can be download from this [link]() according to the original provided train/test text file.) 


## Not finished yet !