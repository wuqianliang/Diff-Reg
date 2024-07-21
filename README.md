<div style="text-align: center;">
  <h2>Diff-Reg: Diffusion Model in Doubly Stochastic Matrix Space for Registration Problem [ECCV2024] [<a href="https://arxiv.org/pdf/2403.19919">arxiv Paper</a>]</h2>
</div>

👀 If you have any questions, please let me (wuqianliang@njust.edu.cn) know~

## Pre-trained Weights
Please look at the [release](https://github.com/wuqianliang/Diff-Reg/releases/tag/DiffRegv1) page for the pre-trained model weights of three experiments.
## Installation
1. Please use the NVIDIA RTX3090 GPU.
2. Utilize the 'Diff-Reg-2d3d/eccv24_2d3d_env.yml', 'Diff-Reg-3dmatch/eccv24_3d_env.yml', 'Diff-Reg-4dmatch/eccv24_4d_env.yml' to install environments.
   
## Data Preparation
Please refer to [Lepard](https://github.com/rabbityl/lepard) and [2D3D-MATR](https://github.com/minhaolee/2D3DMATR).

Our 2D3D registration code  is based on [2D3D-MATR](https://github.com/minhaolee/2D3DMATR), and 3d registration code is based on [Lepard](https://github.com/rabbityl/lepard).

## Training

## Results


## :hearts: Acknowledgement
We thank the respective authors of [Lepard](https://github.com/rabbityl/lepard),[2D3D-MATR](https://github.com/minhaolee/2D3DMATR), [GeoTR](https://github.com/qinzheng93/GeoTransformer),[RoITR](https://github.com/haoyu94/RoITr),[GraphSCNet](https://github.com/qinzheng93/GraphSCNet), and [Vision3D](https://github.com/qinzheng93/vision3d) for their open source code.

## Citation

Please consider citing the following BibTeX entry if you find our work helpful for your research.   

```bibtex
@article{wu2024diff,
  title={Diff-Reg v1: Diffusion Matching Model for Registration Problem},
  author={Wu, Qianliang and Jiang, Haobo and Luo, Lei and Li, Jun and Ding, Yaqing and Xie, Jin and Yang, Jian},
  journal={arXiv preprint arXiv:2403.19919},
  year={2024}
}



