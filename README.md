### [video](https://drive.google.com/file/d/1XIB0MgE1uxi32xiYpwMDjRMmqXvotu79/view?usp=sharing) | [slides](https://docs.google.com/presentation/d/1wZt4OzPpD22AWqFMcoUNqfLbzwPpFqMg/edit?usp=sharing&ouid=116829110622813007518&rtpof=true&sd=true) | [paper](https://arxiv.org/pdf/2210.00174.pdf)
## News
- **(2022/6/20)** We are invited to present our method at the [CVPR 2022 VizWiz Grand Challenge Workshop](https://vizwiz.org/workshops/2022-workshop/)
- **(2022/5/13)** Great! We are the winner of the [ORBIT Few-Shot Object Recognition Challenge](https://eval.ai/challenge/1438/overview)


## Abstract
In this work, we present the winning solution for ORBIT Few-Shot Video Object Recognition Challenge 2022. Built upon the ProtoNet baseline, the performance of our method is improved with three effective techniques. These techniques include the embedding adaptation, the uniform video clip sampler and the invalid frame detection. In addition, we re-factor and re-implement the official codebase to encourage modularity, compatibility and improved performance. Our implementation accelerates the data loading in both training and testing.

## Team Information
**Team name**: canada_goose

<!---
**Institute**:  Noah's Ark Lab, Toronto office, Huawei Canada
-->

**Team members**
- Li Gu
- Zhixiang Chi*
- Huan Liu*
- Yuanhao Yu
- Yang Wang
- Jin Tang

\* denotes equal contribution.

**Contact**
- li.gu@huawei.com
- liuh127@mcmaster.ca

If you use the code in this repo, please cite the following bib entries:

    @article{gu2022improving,
        title={Improving protonet for few-shot video object recognition: Winner of orbit challenge 2022},
        author={Gu, Li and Chi, Zhixiang and Liu, Huan and Yu, Yuanhao and Wang, Yang},
        journal={arXiv preprint arXiv:2210.00174},
        year={2022}
      }

<!-- ## Our proposed method

ProtoNet baseline method does not perform very well on the ORBIT dataset because the few-shot learner cannot build high-quality prototypes at the personalization stage.
To be specific, there are three reasons. Firstly, due to the distribution shift between support and query video sequences (clean vs clutter), using the shared backbone network to extract clip features from both two sets is sub-optimal. Secondly, each user's video frames from different object 
categories usually share similar backgrounds, and even multiple target user-specific objects appear in one frame. Thirdly, there are dramatic appearance changes across each 
support video sequence, and some frames suffer from "object_not_present_issue". Thus, randomly sampled clips from support video sequences will not contribute comprehensive information 
on prototypes. -->

## Our proposed method

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


<!-- 1. We add one transformer encoder block on top of prototypes, a similar idea borrowed from [FEAT, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf). It enforces the interaction among the prototypes to yield more discriminative representations and adapt better to the users in the current episode.  It also helps to reduce the distribution shift by mapping both features to a more similar feature space.
-->

1. During both training and testing, we add one transformer encoder block on prototypes, a similar idea from [FEAT, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf). 
It can make each object's prototype adapted to the specific episode and highlight their most discriminative representation for a specific user. 
Also, the transformer encoder block can map support features (clean) to the space close to the query's (clutter) and help alleviate the distribution shift. 

2. During testing, we replace the random support clip sampler by the uniform sampler to achieve higher temporal coverages. We firstly split each support video sequence into multiple fix-sized and non-overlapped clip candidates following ORBIT codebase, and then evenly split clip candidates into non-overlapped chunks. Each chunk has same number of clip candidates. At last, we sample one clip from each chunk. Figure 2 demonstrates the details.
3. During testing, we apply an edge detector on each sampled support frame and set a empirical threshold to determine whether the frame contains objects. 
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


**step1: Installation**

The following packages are required:
- python 3.8
- pytorch_lightning
- torchmetrics
- pytorch 1.10.0+cu102
- torchvision 0.11.1+cu102
- hydra
- einops

```shell
conda env create -f environment.yml
```

**step 2: Set environment variables** 
```shell
cd your_project_folder_path
export PROJECT_ROOT= your_project_folder_path
export PYTHONPATH= your_project_folder_path
```

**step 3: Download ORBIT dataset**

Please download the dataset from [ORBIT benchmark](https://github.com/microsoft/ORBIT-Dataset) into the folder `your_orbit_dataset_folder_path`

Check that the folder includes train, validation, test sets of video frame images and their annotations, following the format: 
```
/ORBIT_microsoft
|
|--- train/
      |--- P100/
      |--- exercise bench/
      |------------clean/
      |---------------P100--exercise-bench--clean--4ChvjQ3Xzidvq0mCI9lperemxb6D6tCyQS-BG6LS72A/
      |------------------ P100--exercise-bench--clean--4ChvjQ3Xzidvq0mCI9lperemxb6D6tCyQS-BG6LS72A-00001.jpg
      |------------------ P100--exercise-bench--clean--4ChvjQ3Xzidvq0mCI9lperemxb6D6tCyQS-BG6LS72A-00002.jpg
      ...
      
|--- validation/
|--- test/
|--- annotation/ 
      |------ orbit_extra_annotations/
      |------ orbit_train_object_cluster_labels.json
      |------ orbit_validation_object_cluster_labels.json
      |------ orbit_test_object_cluster_labels.json
```
Change the data root path, `data.train_cfg.root = your_orbit_dataset_folder_path`, `data.val_cfg.root = your_orbit_dataset_folder_path` and
`data.test_cfg.root = your_orbit_dataset_folder_path`. [example](https://github.com/Guliisgreat/ORBIT_Challenge_2022_Team_canada_goose/blob/e004c1265681b41ab03b5a249f869c999d3e3249/pytorchlightning_trainer/conf/data/default.yaml#L7)

**step 4: Download checkpoints to reproduce our best testing result**

To reproduce our best result, please download the checkpoint from [FEAT_data_aug](https://drive.google.com/drive/folders/1BhxylCNmAt6dQ-nHXw4Orv62kBiZkHAH?usp=sharing) into the folder `checkpoints`

Then, please change the `load_pretrained` in `pytorchlightning_trainer/cfg/train/with_lite_test`.[here](https://github.com/Guliisgreat/ORBIT_Challenge_2022_Team_canada_goose/blob/ab9c237896482c75861179588e7ec0a2afb7acbe/pytorchlightning_trainer/conf/train/with_lite_test.yaml#L19)
## Code Structures

To reproduce our experiments, please use **run.py**. There are four parts in the code.

 - `src`: It contains all source codes. 
   - `src/official_orbit`: codes copied from orbit original codebase
   - `src/data`: our re-implemented data pipeline on ORBIT dataset
   - `src/tranform`: transform functions commonly used in episodic learning (meta learning) 
   - `src/learner`:  few shot learning methods (e.g. FEAT)
 - `pytorchlightning_trainer`: we follow the **_pytorchlightning-hydra_** template to make codes and experiments **structured**, **readable** and **reproducible**. 
[why use lightning-hydra](https://github.com/ashleve/lightning-hydra-template)
   - `pytorchlightning_trainer/callbacks`:  Lightning callbacks
   - `pytorchlightning_trainer/datamodule`: Lightning datamodules
   - `pytorchlightning_trainer/module`: Lightning modules
   - `pytorchlightning_trainer/conf`: [Hydra](https://hydra.cc/) configuration files
 - `checkpoints`: please download checkpoints from GoogleDrive Links in Table.
 - `logs/tb_logs`: the training logging results (tensorboard) and testing results (orbit_submission.json) will be automatically saved



## Reproduce our best result on ORBIT Challenge 2022 Leaderboard
### Training with our re-implemented data pipeline
```shell
python run.py  
data=default_use_data_aug 
model=feat_with_lite 
train=with_lite_train
train.exp_name="reproduce_our_best_model_training_with_data_augment"
```
Then, you can find tensorboard logs and checkpoints in `your_project_folder_path/logs/tb_logs/"reproduce_our_best_model_training_with_data_augment/version_x`

### Testing with our re-implemented data pipeline
```shell
python run.py
data=test_support_sampler_uniform_fixed_chunk_size_10 
model=feat_with_lite_video_post 
train=with_lite_test 
train.exp_name="reproduce_our_leaderboard_testing_result"
```
Then, you can find testing results with submission format in `your_project_folder_path/logs/tb_logs/reproduce_our_leaderboard_testing_result/testing_per_video_results/orbit_submission.json`

## Extra contributions to code quality
### Refactor the data pipeline
We refactor and re-implement the data pipeline of the original ORBIT codebase to encourage modularity, compatibility 
and performance 
1. **Modularity**
   - We decouple the logic of object category sampling, video sequence (instance) sampling, video clip sampling, 
   video frame image loading and tensor preparing from one deeper class `data.datasets.ORBITDataset` into 
   multiple independent shallow classes.
     - `src.data.sampler`: To sample object categories from each user, and video instances from each object category 
     - `src.clip_sampler`: To sample clips from each video instance 
     - `src.video`: To load video frames using multi threads
     - `src.orbit_few_shot_video_classification`: To assemble above components and transform images to tensors. 
   - It provides independent components. These components can be used in plug-and-play and mix-and-match manners.  For example, we replace the original random clip sampler 
   `src.clip_sampler.RandomMultiClipSampler` by the uniform sampler 
   `src.clip_sampler.UniformFixedChunkSizeMultiClipSampler`. [](and enable to build high quality prototypes in testing.)
     
2. **Compatibility**
   - It is designed to be interoperable with other Pytorch standard domain specific libraries (torchvision), and
   their highly-optimized modules and functions can be use in the processing of ORBIT dataset. 
   - Its API is kept similar with common usage in standard supervised learning: `torch.utils.data.Dataset` to 
   prepare each mini-batch of episodes, and `torch.utils.data.DataLoader` to handle batching, shuffling and 
   prefetching with multi CPU workers 

4. **Performance**
   - The data pipeline of the original codebase cannot maintain the GPU utility 100%, and thus the data preparation
   becomes the bottleneck.
   - To optimize I/O, we introduce the multithreading to hide the latency of loading hundreds of images from disk 
   in each episode. The speed benchmark is in the following tables
   - Testing Machine
     - CPU: Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz, 72 cores
     - GPU: Nvidia V-100 32G, 1 card 

#### Testing speed
We compare the speed of data pipelines, which count the total time of preparing 300 video sequence tensors 
from 17 users in test set.
* The data preparation consists of several steps, such as sampling objects, sampling instances for both query and support sets, sampling support 
video clips, loading clip frame images and converting image arrays into tensors.

|                         | num workers | num threads | total time (s) |
|-------------------------|-------------|-------------|----------------|
| original ORBIT codebase | 4           | 1           | 233            |
| ours                    | 4           | 4           | 201 (1.15x)    |
| ours                    | 4           | 16          | 152 (1.53x)    |
| ours                    | 8           | 16          | 86  (2.70x)    |

#### Training speed
We compare the speed of data pipelines, which average the time after preparing tenors of 100 episodes in train set.

|                         | num workers | num threads | time per episode (s) |
|-------------------------|-------------|-------------|----------------------|
| original ORBIT codebase | 4           | 1           | 2.61                 |
| ours                    | 4           | 4           | 2.20 (1.18x)         |
| ours                    | 4           | 16          | 1.08 (2.41x)         |
| ours                    | 8           | 16          | 0.94 (2.77x)         |

### To Do 
1. To accelerate and parallelize few-shot learners' training, we will replace original gradient accumulation 
techniques with `torch.distributed` package using multiple GPUs.
2. We will modularize the original class of few-shot-learner: backbone network, normalization layer, adaptation module, 
LITE usage, classifier.  


## Acknowledgment

We thank the following repos providing helpful components/functions in our work.

- [FEAT](https://github.com/Sha-Lab/FEAT/tree/47bdc7c1672e00b027c67469d0291e7502918950)

- [PytorchVideo](https://github.com/facebookresearch/pytorchvideo)

- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

- [ORBIT Dataset](https://github.com/microsoft/ORBIT-Dataset)

