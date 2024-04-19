# ReCODE

>ReCODE

>Modeling Repeat Consumption with Neural ODE

This is the official implementation for the SIGIR 2024 paper "ReCODE: Modeling Repeat Consumption with Neural ODE".

## Introduction

ReCODE is a novel model-agnostic framework that utilizes neural ODE to model repeat consumption. And ReCODE seamlessly integrates with various existing recommendation models, including collaborative-based and sequential-based models, making it easily applicable in different scenarios. Moreover, ReCODE achieves great performance on two public datasets: MMTD and Nowplaying-RS.

## Citation

**Please cite our paper if you use our codes. Thanks!**
```
@inproceedings{wang2019modeling,
  title={ReCODE: Modeling Repeat Consumption with Neural ODE},
  author={Sun, Haodai and Qu, Changle and Chen, Sirui and Zhang, Xiao and Xu, Jun},
  booktitle={The ACM Special Interest Group on Information Retrieval},
  year={2024},
  organization={ACM}
}
```

## Quick Start

Please run the shell file "run.sh" to run our proposed ReCODE and other baselines.

> ./run.sh

## Environment

Our experimental environment is shown below:

```
numpy version: 1.23.3
pandas version: 1.4.4
torch version: 1.8.0
```
## Datasets

- **MMTD**: The original data and description is avaliable [here](http://www.cp.jku.at/datasets/MMTD/).
- **Nowplaying-RS**: The original data and description is avaliable [here](https://zenodo.org/record/3247476#.Yhnb7ehBybh).

## Usage		

> python main.py --gpu 0 --model_name MF --emb_size 32 --lr 5e-4 --l2 1e-7 --dataset "MMTD"

You can specify the gpu id, the model name, the used dataset by cmd line arguments. The detailed introduction of the hyper-parameters can be seen in the following implementation details, and you are highly encouraged to read the paper to better understand the effects of some key hyper-parameters.


## Implementation Details
We use the Euler ODE solver from the [torchdiffeq](https://github.com/rtqichen/torchdiffeq) package. The Encoder, $f_\text{ODE}$, and Decoder of ReCODE are constructed using two-layer MLPs with hidden size set to $64$. To ensure a fair comparison, we set the embedding size of all methods to $32$ and use a batch size of $512$. The Adam optimizer is employed to train all models, and we carefully tune the learning rate among {1e-3, 5e-4,1e-4,5e-5,1e-5}, as well as the weight decay among {1e-5, 1e-6, 1e-7}. For sequential-based methods, we set the maximum length of historical interactions to $20$. 

