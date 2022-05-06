## Team Information
**Team name**: canada_goose

**Institute**:  Noah's Ark Lab, Toronto office, Huawei Canada

**Team members**
- Li Gu
- Zhixiang Chi
- Huan Liu
- Yuanhao Yu
- Yang Wang

**Contact**
- li.gu@huawei.com
- liuh127@mcmaster.ca

<!-- ## Our proposed method

ProtoNet baseline method does not perform very well on the ORBIT dataset because the few-shot learner cannot build high-quality prototypes at the personalization stage.
To be specific, there are three reasons. Firstly, due to the distribution shift between support and query video sequences (clean vs clutter), using the shared backbone network to extract clip features from both two sets is sub-optimal. Secondly, each user's video frames from different object 
categories usually share similar backgrounds, and even multiple target user-specific objects appear in one frame. Thirdly, there are dramatic appearance changes across each 
support video sequence, and some frames suffer from "object_not_present_issue". Thus, randomly sampled clips from support video sequences will not contribute comprehensive information 
on prototypes. -->

## Our proposed method
### [video](https://mcmasteru365-my.sharepoint.com/:v:/g/personal/liuh127_mcmaster_ca/EdZHIOxqoFhAn5UUj85Q2fQBZd6YYKFsuemTQOtw3X6WDA?e=XNB53A)

ProtoNet aims to produce class prototypes from the support data samples at the personalization stage, and the query data samples can be classified by directly comparing their embeddings with the prototypes using a similarity metric at the recognization stage. Due to the characteristics of the ORBIT-dataset, there are several reasons that hinder the generation of high-quality prototypes. First, due to the distribution shift between support and query videos (clean vs. clutter), using the same backbone to extract clip features from both two sets is sub-optimal. Second, each user's videos are collected from limited scenes, which results in similar background or multiple target user-specific objects [issue](https://github.com/microsoft/ORBIT-Dataset/issues/4). Thirdly, there are dramatic appearance changes across each support video sequence, and some frames suffer from "object_not_present_issue". Thus, randomly sampled clips from support video sequences will not contribute comprehensive information 
on prototypes.

To make the few-shot learner build high-quality prototypes at the personalization stage, we develop three techniques on top of the ProtoNet baseline method. 
The pipeline is shown in Figure 1.

<p align = "center">
<img src = "https://github.com/Guliisgreat/ORBIT_Challenge_2022_Team_canada_goose/blob/main/docs/orbit_pipline.JPG">
</p>
<p align = "center">
Fig.1 Overview of our proposed method. 
</p>

<!-- 1. During both training and testing, we add one transformer encoder block on prototypes, a similar idea from [FEAT, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf). 
It can make each object's prototype adapted to the specific episode and highlight their most discriminative representation for a specific user. 
Also, the transformer encoder block can map support features (clean) to the space close to the query (clutter) and help alleviate the distribution shift. -->

1. We add one transformer encoder block on top of prototypes, a similar idea borrowed from [FEAT, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf). It enforces the interaction among the prototypes to yield more discriminative representations and adapt better to the users in the current episode.  It also helps to reduce the distribution shift by mapping both features to a more similar feature space.

2. During testing, we replace the random support clip sampler with the uniform sampler to make sure higher temporal coverages. We firstly split each support video sequence into multiple fix-sized and non-overlapped clips following ORBIT codebase, and then further evenly split clips into non-overlapped chunks, where each chunk has same number of clips. At last, we sample one clip from each chunk. Figure 2 demonstrates the details.
3. During testing, we apply an edge detector on each sampled support frame and use a hand-tuned threshold to determine whether the frame contains objects. 
Specifically, if more than half of the frames from one clip are identified with "object_not_present_issue", that clip will be removed.

<p align = "center">
<img src = "https://github.com/Guliisgreat/ORBIT_Challenge_2022_Team_canada_goose/blob/main/docs/uniform_sampler2.JPG">
</p>
<p align = "center">
Fig.2 - our uniform support clips sampler 
</p>


## Our ORBIT  Results
Experimental results on ORBIT datasets with EfficienNet_B0 backbone and images with 224 x 224 resolution.

|                                     | random seed | support clip sampler | data augment | missing object detector | Frame Accuracy | Checkpoint |
|-------------------------------------|-------------|----------------------|--------------|-------------------------|----------------|------------|
| ProtoNet baseline (NeurIPS21 paper) |             |                      |              |                         | 66.3           |            |
| ProtoNet baseline (ORBIT codebase)  | 42          | Random               |              |                         | 63.73          | [ProtoNet_baseline](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_protonets_efficientnetb0_224_lite.pth)          |
| ProtoNet baseline (ours codebase)   | 42          | Random               |              |                         | 66.27          | [ProtoNet_baseline](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_protonets_efficientnetb0_224_lite.pth)          |
| FEAT baseline                       | 42          | Random               |              |                         | 70.13 (+3.86)  | [FEAT_baseline](https://drive.google.com/drive/folders/1juvUjdF-h65z_372hKIJKdW3OkzTG1Re?usp=sharing)          |
| FEAT + Uniform                      | 42          | Uniform              |              |                         | 70.69 (+4.42)  | [FEAT_baseline](https://drive.google.com/drive/folders/1juvUjdF-h65z_372hKIJKdW3OkzTG1Re?usp=sharing)           |
| FEAT+Aug+Uniform                    | 42          | Uniform              |       ✓      |                         | 71.57 (+5.3)   | [FEAT_data_aug](https://drive.google.com/drive/folders/1BhxylCNmAt6dQ-nHXw4Orv62kBiZkHAH?usp=sharing)           |
| Our best                            | 42          | Uniform              |       ✓      |            ✓            | 71.69 (+5.42)  | [FEAT_data_aug](https://drive.google.com/drive/folders/1BhxylCNmAt6dQ-nHXw4Orv62kBiZkHAH?usp=sharing)           |
| Our best                            | 20          | Uniform              | ✓            | ✓                       | 71.78          | [FEAT_data_aug](https://drive.google.com/drive/folders/1BhxylCNmAt6dQ-nHXw4Orv62kBiZkHAH?usp=sharing)           |

## Prerequisites


Install library dependencies:
```shell
conda create --name <your_env_name> --file requirements.txt
```

Set environment variables 
```shell
cd your_project_folder_path
export PROJECT_ROOT= your_project_folder_path
export PYTHONPATH= your_project_folder_path
```

Please download the dataset from [ORBIT benchmark](https://github.com/microsoft/ORBIT-Dataset)
```shell
/ORBIT_microsoft
-- train
-- test
-- validation
-- annotation 
```
To testing our best result, please download our best model checkpoint from [FEAT_data_aug](https://drive.google.com/drive/folders/1BhxylCNmAt6dQ-nHXw4Orv62kBiZkHAH?usp=sharing) into the folder `checkpoints`

Then, please change the `load_pretrained` in `pytorchlightning_trainer/cfg/train/with_lite_test`.[here](https://github.com/Guliisgreat/ORBIT_Challenge_2022_Team_canada_goose/blob/ab9c237896482c75861179588e7ec0a2afb7acbe/pytorchlightning_trainer/conf/train/with_lite_test.yaml#L19)
## Code Structures

To reproduce our experiments, please use **run.py**. There are four parts in the code.

 - `src`: It contains all source codes. 
   - `src/official_orbit`: codes copied from orbit original codebase
   - `src/data`: our re-implemented data pipeline on ORBIT dataset
   - `src/tranform`: transform functions commonly used in episodic learning (meta learning) 
   - `src/learner`: added few shot learning methods (e.g. FEAT)
 - `pytorchlightning_trainer`: we follow the [pytorchlightning](https://www.pytorchlightning.ai/)-hydra template to make codes and experiments structured, readable and reproducible. 
[why use lightning-hydra](https://github.com/ashleve/lightning-hydra-template)
   - `pytorchlightning_trainer/callbacks`:  Lightning callbacks
   - `pytorchlightning_trainer/datamodule`: Lightning datamodules
   - `pytorchlightning_trainer/module`: Lightning modules
   - `pytorchlightning_trainer/conf`: [Hydra](https://hydra.cc/) configuration files
 - `checkpoints`: please downloaded checkpoints from GoogleDrive Links in Table.
 - `logs/tb_logs`: the training logging results (tensorboard) and testing results (orbit_submission.json) will be automatically saved



## Reproduce our best result on ORBIT Challenge 2022 Leaderboard
### Training with our re-implemented data pipeline
```shell
python run.py  
data=default_use_data_aug 
model=feat_with_lite 
train=with_lite_train
train.exp_name="reproduce_our_training_result"
```

### Testing with our re-implemented data pipeline and evaluate with our re-implemented evaluation toolbox
```shell
python run.py
data=test_support_sampler_uniform_fixed_chunk_size_10 
model=feat_with_lite_video_post 
train=with_lite_test 
train.exp_name="reproduce_our_testing_result"
```




## Acknowledgment

We thank the following repos providing helpful components/functions in our work.

- [FEAT](https://github.com/Sha-Lab/FEAT/tree/47bdc7c1672e00b027c67469d0291e7502918950)

- [PytorchVideo](https://github.com/facebookresearch/pytorchvideo)

- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

- [ORBIT Dataset](https://github.com/microsoft/ORBIT-Dataset)

