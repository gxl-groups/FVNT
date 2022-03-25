# FVNT
## [DataSet](https://pan.baidu.com/s/1ik27IF56ZK50bUmuu3WTCg?pwd=3m9y)  [Model](https://pan.baidu.com/s/1eHe85WQqhtwcmmeNR1V4AQ?pwd=u4v5)
> Download the dataset and place it in the ./dataset;
> Download the model and place it in the ./model.
## Prerequisites
- env1: 
    - torch 1.4.0 + cu92
    - RTX 2080ti
- env2:
    - torch 1.1.0
    - TitanXP
## Getting Started

### Installing
-  **Install Deformable Conv**
    
    > sh ./Deformable/make.sh

### Train
-  **Train Stage 1**
   
    > python train_viton_stage_1.py
-  **Train Stage 2**
   
    > python train_viton_stage_2.py
-  Train Stage 3
   
    > python train_viton_stage_3.py

### Test
- For some reason, in our experiments, we trained the model of Stage1 on env1 and the models of Stage2 and Stage3 on env2. We tried to get the final results of the pre-trained model of Stage1 in env2 environment with Stage2 and Stage3, but found that the results would be worse due to the fact that the training environment of Stage1 is not env2. So, we first got the parsing needed for testing in env1 and saved it for subsequent use. These data can be found in our dataset.
- Test cross pair:To show the qualitative results, we use the same pair as cpvton.
  
    > python test.py 
- Test self pair:To calculate the various metrics, we use the same pair as cpvton+.
  
    > python test.py --file_path test_pairs_self.txt --generate_parsing generate_parsing_self

​    





