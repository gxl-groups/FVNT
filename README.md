# A Flow-based Generative Network for Photo-Realistic Virtual Try-On

![image1.png](https://github.com/gxl-groups/FVNT/blob/main/pics/1.jpg)

In this paper,we propose a novel Flow-based Virtual Try-on Network (FVTN). It consists of three modules. Firstly, the Parsing Alignment Module (PAM) aligns the source clothing to the target person at the semantic level by predicting a semantic parsing map. Secondly, the Flow Estimation Module (FEM) learns a robust clothing deformation model by estimating multi-scale dense flow fields in an unsupervised fashion. Thirdly, the Fusion and Rendering Module (FRM) synthesizes the final try-on image by effectively integrating the warped clothing features and human body features.

# Prerequisites

- Enviroment1(env1): 
  - pytorch 1.4.0
  - numpy
  - torchvision
  - RTX 2080ti
- Enviroment2(env2):
  - pytorch 1.1.0
  - numpy
  - torchvision
  - Deformable Conv
  - TitanXP

# Getting Started

## Installing

- Install Deformable Conv `sh ./Deformable/make.sh` 

## Data Preperation

We provide our **dataset files**  for convience. Download the models below and put it under `dataset/`

- Download the VITON dataset from [here](https://pan.baidu.com/s/1ik27IF56ZK50bUmuu3WTCg?pwd=3m9y) .

## Train the model

- Train PAM `python train_viton_stage_1.py`
- Train FEM `python train_viton_stage_2.py`
- Train FRM `python train_viton_stage_3.py` 

## Test the model

- We trained the model of PAM on env1 and the models of FEM and FRM on env2. We first got the parsing needed for testing in env1 and saved it for subsequent use. These semantic parsing maps can be found in this [dataset](https://pan.baidu.com/s/1ik27IF56ZK50bUmuu3WTCg?pwd=3m9y).
- Test cross pair:To show the qualitative results, we use the same pair as cpvton.`python test.py `
- Test self pair:To calculate the various metrics, we use the same pair as cpvton+.`python test.py --file_path test_pairs_self.txt --generate_parsing generate_parsing_self`

## Pretrained models

Download the **models** below and put it under `model/`

- Download pretrained models from [here](https://pan.baidu.com/s/1eHe85WQqhtwcmmeNR1V4AQ?pwd=u4v5).

## Example Results

![image2.png](https://github.com/gxl-groups/FVNT/blob/main/pics/2.jpg)
