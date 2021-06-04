# FS-IL

## :wrench: Dependencies and Installation

- Python = 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch = 1.8.1](https://pytorch.org/)
- NVIDIA GPU RTX3090 + [CUDA 11.0](https://developer.nvidia.com/cuda-downloads)

Note that FS-IL is only tested in Ubuntu, and may be not suitable for Windows. You may try [Windows WSL with CUDA supports](https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-cuda-in-wsl) :-) (It is now only available for insider build with Fast ring).

## :computer: Train and Incremental Procdure

- **Training and incremental testing commands**: Please see **[TrainTest.md](docs/TrainTest.md)** for the basic usage.

## Datasets
We evaluate our system in several datasets, including ```CUB-200-2011, CIFAR100, miniImageNet```.
Please download [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) ,  [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) and miniImageNet.


