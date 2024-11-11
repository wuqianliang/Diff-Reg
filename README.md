<div style="text-align: center;">
  <h2>[ECCV2024] Diff-Reg: Diffusion Model in Doubly Stochastic Matrix Space for Registration Problem [<a href="https://arxiv.org/pdf/2403.19919">Arxiv</a>|<a href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08193.pdf">ECCV2024</a>]</h2>
</div>

👀 If you have any questions, please let me (wuqianliang@njust.edu.cn) know~

## Installation
`1. Please use the NVIDIA TITAN RTX or NVIDIA GeForce RTX 3090 GPU !! **If you switch to an RTX 4090 or a higher version GPU, you will need to re-train the model. We have test the Diff-Reg-4dmatch on the RTX 4090 GPU.**`

`2. Please utilize commands 'conda env create -f Diff-Reg-2d3d/eccv24_2d3d_env.yml', 'conda env create -f  Diff-Reg-3dmatch/eccv24_3d_env.yml', and 'conda env create -f Diff-Reg-4dmatch/eccv24_4d_env.yml' to install environments for three tasks.`

## Pre-trained Weights

Please look at the [release](https://github.com/wuqianliang/Diff-Reg/releases/tag/DiffRegv1) page for the pre-trained model weights of three experiments.

## Data Preparation && Training

Our 2D3D registration code is mainly based on [2D3D-MATR](https://github.com/minhaolee/2D3DMATR), and our 3d registration code is based on [Lepard](https://github.com/rabbityl/lepard). Please refer to [Lepard](https://github.com/rabbityl/lepard) and [2D3D-MATR](https://github.com/minhaolee/2D3DMATR).

## Inference
For 3DMatch and 4DMatch:

`python main.py --config configs/test/3dmatch.yaml`

`python main.py --config configs/test/4dmatch.yaml --thr=0.55 # test_epoch=22`

For 2D-3D registration:

`cd experiments/2d3dmatr.rgbdv2.stage4.level3.stage1; sh eval.sh 26`

## Results
#### Quantitative results on the 4DMatch and 4DLoMatch benchmarks. The best results are highlighted in **bold**, and the second-best results are underlined.

| Category         | Method                          | 4DMatch NFMR↑ | 4DMatch IR↑ | 4DLoMatch NFMR↑ | 4DLoMatch IR↑ |
|------------------|---------------------------------|---------------|-------------|-----------------|---------------|
| **Scene Flow**   | PointPWC [Wu et al. 2019]       | 21.60         | 20.0        | 10.0            | 7.20          |
|                  | FLOT [Puy et al. 2020]          | 27.10         | 24.90       | 15.20           | 10.70         |
| **Feature Matching** | D3Feat [Bai et al. 2020]   | 55.50         | 54.70       | 27.40           | 21.50         |
|                  | Predator [Huang et al. 2021]    | 56.40         | 60.40       | 32.10           | 27.50         |
|                  | Lepard [Li et al. 2022]         | 83.60         | 82.64       | 66.63           | 55.55         |
|                  | GeoTR [Qin et al. 2022]         | 83.20         | 82.20       | 65.40           | 63.60         |
|                  | RoITr [Yu et al. 2023]          | 83.00         | **84.40**   | 69.40           | **67.60**     |
| **DDPM**         | Diff-Reg (Backbone)             | **85.47**     | 81.15       | 72.37           | 59.50         |
|                  | Diff-Reg (steps=1)              | 85.23         | 83.85       | **73.19**       | 65.26         |
|                  | Diff-Reg (steps=20)             | **90.25**     | **87.98**   | **77.15**       | **67.00**     |


#### Quantitative results on the 3DMatch and 3DLoMatch benchmarks. The best results are highlighted in **bold**, and the second-best results are underlined.

| Method      | Reference            | 3DMatch FMR↑ | 3DMatch IR↑ | 3DMatch RR↑ | 3DLoMatch FMR↑ | 3DLoMatch IR↑ | 3DLoMatch RR↑ |
|-------------|----------------------|--------------|-------------|-------------|----------------|---------------|---------------|
| FCGF        | ICCV 2019 [Choy et al.] | 95.20        | 56.90       | 88.20       | 60.90          | 21.40         | 45.80         |
| D3Feat      | CVPR 2020 [Bai et al.] | 95.80        | 39.00       | 85.80       | 69.30          | 13.20         | 40.20         |
| Predator    | CVPR 2021 [Huang et al.] | 96.70        | 58.00       | 91.80       | 78.60          | 26.70         | 62.40         |
| Lepard      | CVPR 2022 [Li et al.] | 97.95        | 57.61       | 93.90       | 84.22          | 27.83         | 70.63         |
| GeoTR       | CVPR 2022 [Qin et al.] | 98.10        | 72.70       | 92.30       | 88.70          | 44.70         | 75.40         |
| RoITr       | CVPR 2023 [Yu et al.] | 98.00        | **82.60**   | 91.90       | 89.60          | **54.30**     | **74.80**     |
| PEAL-3D     | CVPR 2023 [Yu et al.] | **98.50**    | 73.30       | **94.20**   | 87.60          | 49.00         | 79.00         |
| Diff-Reg    | ECCV 2024 [Yu et al.] | 96.28        | 30.92       | **95.00**   | 69.60          | 9.60          | 73.80         |


#### Evaluation results on RGB-D Scenes V2 [Li et al. 2023]. The best results are highlighted in **bold**, and the second-best results are underlined.

| Model                             | Scene-11 | Scene-12 | Scene-13 | Scene-14 | Mean  |
|-----------------------------------|----------|----------|----------|----------|-------|
| **Mean depth (m)**                | 1.74     | 1.66     | 1.18     | 1.39     | 1.49  |
| **Inlier Ratio ↑**                |          |          |          |          |       |
| FCGF-2D3D [Choy et al. 2019]      | 6.8      | 8.5      | 11.8     | 5.4      | 8.1   |
| P2-Net [Choy et al. 2019]         | 9.7      | 12.8     | 17.0     | 9.3      | 12.2  |
| Predator-2D3D [Huang et al. 2021] | 17.7     | 19.4     | 17.2     | 8.4      | 15.7  |
| 2D3D-MATR [Li et al. 2023]        | 32.8     | 34.4     | **39.2** | 23.3     | 32.4  |
| FreeReg [Wang et al. 2023]        | 36.6     | 34.5     | 34.2     | 18.2     | 30.9  |
| **Diff-Reg (dino)**               | 38.6     | 37.4     | **45.4** | _31.6_   | _38.3_|
| Diff-Reg (dino/backbone)          | 44.9     | **49.5** | 38.3     | **33.1** | **41.4** |
| Diff-Reg (dino/steps=1)           | **47.5** | _48.9_   | 32.8     | 22.4     | 37.9  |
| Diff-Reg (dino/steps=10)          | _47.2_   | 48.7     | 32.9     | 22.4     | 37.8  |
| **Feature Matching Recall ↑**     |          |          |          |          |       |
| FCGF-2D3D [Choy et al. 2019]      | 11.10    | 30.40    | 51.50    | 15.50    | 27.10 |
| P2-Net [Choy et al. 2019]         | 48.60    | 65.70    | 82.50    | 41.6     | 59.60 |
| Predator-2D3D [Huang et al. 2021] | 86.10    | 89.20    | 63.90    | 24.30    | 65.90 |
| 2D3D-MATR [Li et al. 2023]        | _98.60_  | _98.00_  | 88.70    | 77.90    | 90.80 |
| FreeReg [Wang et al. 2023]        | 91.90    | 93.40    | **93.10**| 49.60    | 82.00 |
| **Diff-Reg (dino)**               | **100.0**| **100.0**| 89.70    | _81.9_   | _92.9_ |
| Diff-Reg (dino/backbone)          | **100.0**| **100.0**| _92.8_   | **91.2** | **96.0** |
| Diff-Reg (dino/steps=1)           | **100.0**| **100.0**| 88.7     | 76.5     | 91.3  |
| Diff-Reg (dino/steps=10)          | **100.0**| **100.0**| 88.7     | 77.0     | 91.4  |
| **Registration Recall ↑**         |          |          |          |          |       |
| FCGF-2D3D [Choy et al. 2019]      | 26.4     | 41.2     | 37.1     | 16.8     | 30.4  |
| P2-Net [Choy et al. 2019]         | 40.3     | 40.2     | 41.2     | 31.9     | 38.4  |
| Predator-2D3D [Huang et al. 2021] | 44.4     | 41.2     | 21.6     | 13.7     | 30.2  |
| 2D3D-MATR [Li et al. 2023]        | 63.9     | 53.9     | 58.8     | 49.1     | 56.4  |
| FreeReg+Kabsch [Wang et al. 2023] | 38.7     | 51.6     | 30.7     | 15.5     | 34.1  |
| FreeReg+PnP [Wang et al. 2023]    | 74.2     | 72.5     | 54.5     | 27.9     | 57.3  |
| **Diff-Reg (dino)**               | 87.5     | 86.3     | 63.9     | 60.6     | 74.6  |
| Diff-Reg (dino/backbone)          | 79.2     | 86.3     | 75.3     | **71.2** | 78.0  |
| Diff-Reg (dino/steps=1)           | **98.6** | **100.0**| _87.6_   | 66.8     | **88.3**|
| Diff-Reg (dino/steps=10)          | **98.6** | _96.1_   | 83.5     | 63.7     | 85.5  |


#### Evaluation results on 7Scenes [Li et al. 2023]. The best results are highlighted in **bold**, and the second-best results are underlined.

| Model                           | Chess   | Fire   | Heads  | Office | Pumpkin | Kitchen | Stairs | Mean  |
|---------------------------------|---------|--------|--------|--------|---------|---------|--------|-------|
| **Mean depth (m)**              | 1.78    | 1.55   | 0.80   | 2.03   | 2.25    | 2.13    | 1.84   | 1.77  |
| **Inlier Ratio ↑**              |         |        |        |        |         |         |        |       |
| FCGF-2D3D [Choy et al. 2019]    | 34.2    | 32.8   | 14.8   | 26.0   | 23.3    | 22.5    | 6.0    | 22.8  |
| P2-Net [Choy et al. 2019]       | 55.2    | 46.7   | 13.0   | 36.2   | 32.0    | 32.8    | 5.8    | 31.7  |
| Predator-2D3D [Huang et al. 2021]| 34.7    | 33.8   | 16.6   | 25.9   | 23.1    | 22.2    | 7.5    | 23.4  |
| 2D3D-MATR [Li et al. 2023]      | 72.1    | 66.0   | 31.3   | 60.7   | 50.2    | 52.5    | 18.1   | 50.1  |
| **Diff-PnP (dino/backbone)**    | **79.2**| **71.0**| **54.1**| **70.4**| **55.8**| **60.2**| **22.9**| **59.1**|
| Diff-PnP (dino/steps=10)        | _73.3_  | 60.8   | 45.5   | 63.1   | 47.8    | 53.3    | 20.4   | 52.0  |
| **Feature Matching Recall ↑**   |         |        |        |        |         |         |        |       |
| FCGF-2D3D [Choy et al. 2019]    | 99.7    | 98.2   | 69.9   | 97.1   | 83.0    | 87.7    | 16.2   | 78.8  |
| P2-Net [Choy et al. 2019]       | 100.0   | 99.3   | 58.9   | 99.1   | 87.2    | 92.2    | 16.2   | 79.0  |
| Predator-2D3D [Huang et al. 2021]| 91.3    | 95.1   | 76.7   | 88.6   | 79.2    | 80.6    | 31.1   | 77.5  |
| 2D3D-MATR [Li et al. 2023]      | 100.0   | 99.6   | 98.6   | 100.0  | 92.4    | 95.9    | 58.1   | 92.1  |
| **Diff-PnP (dino/backbone)**    | **100.0**| **100.0**| **100.0**| **100.0**| **91.3**| **98.1**| **58.1**| **92.5**|
| Diff-PnP (dino/steps=10)        | **100.0**| 98.5   | 97.3   | **100.0**| 87.8    | 96.8    | 60.8   | 91.6  |
| **Registration Recall ↑**       |         |        |        |        |         |         |        |       |
| FCGF-2D3D [Choy et al. 2019]    | 89.5    | 79.7   | 19.2   | 85.9   | 69.4    | 79.0    | 6.8    | 61.4  |
| P2-Net [Choy et al. 2019]       | 96.9    | 86.5   | 20.5   | 91.7   | 75.3    | 85.2    | 4.1    | 65.7  |
| Predator-2D3D [Huang et al. 2021]| 69.6    | 60.7   | 17.8   | 62.9   | 56.2    | 62.6    | 9.5    | 48.5  |
| 2D3D-MATR [Li et al. 2023]      | 96.9    | 90.7   | 52.1   | 95.5   | 80.9    | 86.1    | 28.4   | 75.8  |
| **Diff-PnP (dino/backbone)**    | **100.0**| 94.0   | 90.4   | 99.3   | 81.2    | 94.6    | 27.0   | 83.8  |
| Diff-PnP (dino/steps=10)        | 99.3    | 94.3   | 91.8   | 99.1   | 79.9    | 91.8    | 25.7   | 83.1  |

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



