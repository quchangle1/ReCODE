# ReCODE

This is the official implementation for the SIGIR 2024 paper "ReCODE: Modeling Repeat Consumption with Neural ODE".

## Citation

If you find our code or work useful for your research, please cite our work.
```
@inproceedings{dai2024recode,
  title={ReCODE: Modeling Repeat Consumption with Neural ODE},
  author={Dai, Sunhao and Qu, Changle and Chen, Sirui and Zhang, Xiao and Xu, Jun},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2024}
}
```

## Introduction

ReCODE is a novel model-agnostic framework that utilizes neural ODE to model repeat consumption. Moreover, ReCODE seamlessly integrates with various existing recommendation models, including collaborative-based and sequential-based models, making it easily applicable in different scenarios.



## Quick Start

Please run the shell file "run.sh" to run our proposed ReCODE and other baselines.

> ./run.sh

## Environment

Our experimental environment is shown below:

```
torch version: 1.8.0
```
## Datasets

- **MMTD**: The original data and description is avaliable [here](http://www.cp.jku.at/datasets/MMTD/).
- **Nowplaying-RS**: The original data and description is avaliable [here](https://zenodo.org/record/3247476#.Yhnb7ehBybh).

## Usage		

> python main.py --gpu 0 --model_name MF --emb_size 32 --lr 5e-4 --l2 1e-7 --dataset "MMTD"

You can specify the gpu id, the model name, the used dataset by cmd line arguments. 

## Reference

Our implementations and experiments are conducted based on [ReChorus](https://github.com/THUwangcy/ReChorus) benchmark. We use the Euler ODE solver from the [torchdiffeq](https://github.com/rtqichen/torchdiffeq) package.
