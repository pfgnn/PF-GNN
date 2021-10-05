# PF-GNN

This repository is the official implementation of [PF-GNN: Differentiable particle filtering based approximation of universal graph representations](). 

## Requirements
This code has been developed and tested on Nvidia-RTX-2080Ti GPU with:
1. Pytorch-1.9.0
2. Pytorch-geometric-1.9.0
3. Cuda 10.1/10.2

Other higher versions will probably work as well. Make relevant changes during installation for other versions

If you don't have Anaconda for python 3, install it from [here](https://docs.anaconda.com/anaconda/install/linux/)

Make sure cuda and cuda-Toolkit is installed. Note down the CUDA version in your machine.
```cudaversion
cat /usr/local/cuda/version.txt
nvcc --version
```

To install requirements for cuda version 10.1:

```setup
conda create --name torch_test python=3.6

conda activate torch_test

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install pyg -c pyg -c conda-forge


```
If you encounter problems, refer to pytorch-geometric installation page [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). The code should work with standard installation as well. 


## Training and Evaluation

To train and evaluate, run:

```eval
Regression task:
ZINC: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7;python PF-GNNIR-zinc.py --num_particles=8 --depth=3 --batch_size=512 --dim=150 --parallel

Classification task: export CUDA_VISIBLE_DEVICES=0;python PF-GNNIR-triangles.py --depth=4 --num_particles=24 --dim=128 --batch_size=128


```


This code is based on [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric)
