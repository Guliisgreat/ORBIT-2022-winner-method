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

## Our proposed method
ProtoNet baseline method does not perform very well on the ORBIT dataset because the few-shot learner cannot build high-quality prototypes at the personalization stage.
To be specific, there are three reasons. Firstly, due to the distribution shift between support and query video sequences (clean vs clutter), using the shared backbone network to extract clip features from both two sets is sub-optimal. Secondly, each user's video frames from different object 
categories usually share similar backgrounds, and even multiple target user-specific objects appear in one frame. Thirdly, there are dramatic appearance changes across each 
support video sequence, and some frames suffer from "object_not_present_issue". Thus, randomly sampled clips from support video sequences will not contribute comprehensive information 
on prototypes.

To make the few-shot learner build high-quality prototypes at the personalization stage, we develop three techniques on top of the ProtoNet baseline method. 
The pipeline is shown in Figure 1.

<p align = "center">
<img src = "https://github.com/Guliisgreat/ORBIT_Challenge_2022_Team_canada_goose/blob/main/docs/orbit_pipline.JPG">
</p>
<p align = "center">
Fig.1 - our proposed method 
</p>

1. During both training and testing, we add one transformer encoder block on prototypes, a similar idea from [FEAT, CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ye_Few-Shot_Learning_via_Embedding_Adaptation_With_Set-to-Set_Functions_CVPR_2020_paper.pdf). 
It can make each object's prototype adapted to the specific episode and highlight their most discriminative representation for a specific user. 
Also, the transformer encoder block can map support features (clean) to the space close to the query (clutter) and help alleviate the distribution shift.

2. During testing, we replace the random support clip sampler with the uniform sampler to make sure higher temporal coverages. To follow the common sampling technique in the video
understanding task [SlowFast, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf), 
we evenly split the video sequence into multiple fix-sized and non-overlapped chunks and sample one clip from each chunk. Details are shown in Figure 2
3. During testing, we apply an edge detector on each sampled support frame and use a hand-tuned threshold to determine whether the frame contains objects. 
Specifically, if more than half of the frames from one clip are identified with "object_not_present_issue", that clip will be removed.

<p align = "center">
<img src = "https://github.com/Guliisgreat/ORBIT_Challenge_2022_Team_canada_goose/blob/main/docs/uniform_sampler.JPG">
</p>
<p align = "center">
Fig.2 - our uniform support clips sampler 
</p>


[//]: # (We propose a novel ...)

[//]: # ()
[//]: # (Several challenges we found from ORBIT dataset)

[//]: # (1. Domain shift between clean &#40;support set&#41; and clutter &#40;query set&#41; video sequences)

[//]: # (2. Videos collected from same user &#40;episode&#41; have similar backgrounds, where even some frames include multiple target objects &#40;issue link&#41;)

[//]: # (3. The appearance of each frame has a dramatic change across the video sequence )

[//]: # (4. Some frames have no object present and thus cannot contribute strong information to calculating prototypes  )

[//]: # ()
[//]: # (**Solution**: )

[//]: # (1. To highlight the most discriminative representation for a user, we apply a transformer encoder layer on top of original calculated prototypes to enable strong co-adaptation)

[//]: # (of each object &#40;Idea borrowed from FEAT&#41;)

[//]: # (2. Introduce data augmentation techniques)

[//]: # (3. To contribute high quality embeddings for prototypes , the sampled support clips from each video sequence need to have strong temporal coverage. Also, we hypothesis that the longer)

[//]: # (video sequence will contribute more information. Thus, we introduce our uniform clip sampler...)

[//]: # (4. To avoid sampled support clips have non-object, we apply the edge detector on each sampled support frames. If over half of frames from one support clip have non-object, )

[//]: # (that clip will be removed  )
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (Highlights &#40;71.5&#41; &#40;Required&#41; feat + uniform_fixe_chunk_20 )

[//]: # (1. refactored data pipline --> Introduce more episodes diversity + data augmenetation &#40;engineering contribution&#41;)

[//]: # (2. Replace ProtoNet with FEAT &#40;FSL: visualization + novelty&#41;)

[//]: # (3. Freeze BN layer, because BN performs poorly when i.i.d assumption violated;)

[//]: # (4. testing support clips sampler &#40;video understanding: temporal coverage&#41;)

[//]: # (5. canny detector &#40;optional&#41;)

## Our ORBIT  Results
Experimental results on ORBIT datasets with EfficienNet_B0 backbone and images with 224 x 224 resolution. Please check the detailed experiment setups and hyperparameters in [] 

|                                     | random seed | support clip sampler | data augment | missing object detector | Frame Accuracy | Checkpoint |
|-------------------------------------|-------------|----------------------|--------------|-------------------------|----------------|------------|
| ProtoNet baseline (NeurIPS21 paper) |             |                      |              |                         | 66.3           |            |
| ProtoNet baseline (ORBIT codebase)  | 42          | Random               |              |                         | 63.73          | [ProtoNet_baseline](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_protonets_efficientnetb0_224_lite.pth)          |
| ProtoNet baseline (ours codebase)   | 42          | Random               |              |                         | 66.27          | [ProtoNet_baseline](https://github.com/microsoft/ORBIT-Dataset/raw/master/checkpoints/orbit_cluve_protonets_efficientnetb0_224_lite.pth)          |
| FEAT baseline                       | 42          | Random               |              |                         | 70.13 (+3.86)  | [FEAT_baseline](https://drive.google.com/drive/folders/1juvUjdF-h65z_372hKIJKdW3OkzTG1Re?usp=sharing)          |
| FEAT + Uniform                      | 42          | Uniform              |              |                         | 70.69 (+4.42)  | [FEAT_baseline](https://drive.google.com/drive/folders/1juvUjdF-h65z_372hKIJKdW3OkzTG1Re?usp=sharing)           |
| FEAT+Aug+Uniform                    | 42          | Uniform              |       ✓      |                         | 71.57 (+5.3)   | [FEAT_data_aug](https://drive.google.com/drive/folders/1BhxylCNmAt6dQ-nHXw4Orv62kBiZkHAH?usp=sharing)           |
| Our best                            | 42          | Uniform              |       ✓      |            ✓            | 71.69 (+5.42)  | [FEAT_data_aug](https://drive.google.com/drive/folders/1BhxylCNmAt6dQ-nHXw4Orv62kBiZkHAH?usp=sharing)           |


[//]: # (## Prerequisites)

[//]: # ()
[//]: # (The following packages are required to run the scripts:)

[//]: # ()
[//]: # (- [PyTorch-1.4 and torchvision]&#40;https://pytorch.org&#41;)

[//]: # ()
[//]: # (- Package [tensorboardX]&#40;https://github.com/lanpa/tensorboardX&#41;)

[//]: # ()
[//]: # (- Dataset: please download the dataset and put images into the folder data/[name of the dataset, miniimagenet or cub]/images)

[//]: # ()
[//]: # (- Pre-trained weights: please download the [pre-trained weights]&#40;https://drive.google.com/open?id=14Jn1t9JxH-CxjfWy4JmVpCxkC9cDqqfE&#41; of the encoder if needed. The pre-trained weights can be downloaded in a [zip file]&#40;https://drive.google.com/file/d/1XcUZMNTQ-79_2AkNG3E04zh6bDYnPAMY/view?usp=sharing&#41;.)

[//]: # ()
[//]: # ()
[//]: # (## Code Structures)

[//]: # (To reproduce our experiments, please use **run.py**. There are four parts in the code.)

[//]: # ( - `model`: It contains the main files of the code, including the few-shot learning trainer, the dataloader, the network architectures, and baseline and comparison models.)

[//]: # ( - `data`: Images and splits for the data sets.)

[//]: # ( - `saves`: The pre-trained weights of different networks.)

[//]: # ( - `checkpoints`: To save the trained models.)

[//]: # ()
[//]: # (## Reproduce our result on ORBIT Challenge 2022 Leaderboard)

[//]: # (Please use **train_fsl.py** and follow the instructions below. FEAT meta-learns the embedding adaptation process such that all the training instance embeddings in a task is adapted, based on their contextual task information, using Transformer. The file will automatically evaluate the model on the meta-test set with 10,000 tasks after given epochs.)


## Acknowledgment

We thank the following repos providing helpful components/functions in our work.

- [FEAT](https://github.com/Sha-Lab/FEAT/tree/47bdc7c1672e00b027c67469d0291e7502918950)

- [PytorchVideo](https://github.com/facebookresearch/pytorchvideo)

- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

- [ORBIT Dataset](https://github.com/microsoft/ORBIT-Dataset)


[//]: # (## Arguments)

[//]: # (The train_fsl.py takes the following command line options &#40;details are in the `model/utils.py`&#41;:)

[//]: # ()
[//]: # (**Task Related Arguments**)

[//]: # (- `dataset`: Option for the dataset &#40;`MiniImageNet`, `TieredImageNet`, or `CUB`&#41;, default to `MiniImageNet`)

[//]: # ()
[//]: # (- `way`: The number of classes in a few-shot task during meta-training, default to `5`)

[//]: # ()
[//]: # (- `eval_way`: The number of classes in a few-shot task during meta-test, default to `5`)

[//]: # ()
[//]: # (- `shot`: Number of instances in each class in a few-shot task during meta-training, default to `1`)

[//]: # ()
[//]: # (- `eval_shot`: Number of instances in each class in a few-shot task during meta-test, default to `1`)

[//]: # ()
[//]: # (- `query`: Number of instances in each class to evaluate the performance during meta-training, default to `15`)

[//]: # ()
[//]: # (- `eval_query`: Number of instances in each class to evaluate the performance during meta-test, default to `15`)

[//]: # ()
[//]: # (**Optimization Related Arguments**)

[//]: # (- `max_epoch`: The maximum number of training epochs, default to `200`)

[//]: # ()
[//]: # (- `episodes_per_epoch`: The number of tasks sampled in each epoch, default to `100`)

[//]: # ()
[//]: # (- `num_eval_episodes`: The number of tasks sampled from the meta-val set to evaluate the performance of the model &#40;note that we fix sampling 10,000 tasks from the meta-test set during final evaluation&#41;, default to `200`)

[//]: # ()
[//]: # (- `lr`: Learning rate for the model, default to `0.0001` with pre-trained weights)

[//]: # ()
[//]: # (- `lr_mul`: This is specially designed for set-to-set functions like FEAT. The learning rate for the top layer will be multiplied by this value &#40;usually with faster learning rate&#41;. Default to `10`)

[//]: # ()
[//]: # (- `lr_scheduler`: The scheduler to set the learning rate &#40;`step`, `multistep`, or `cosine`&#41;, default to `step`)

[//]: # ()
[//]: # (- `step_size`: The step scheduler to decrease the learning rate. Set it to a single value if choose the `step` scheduler and provide multiple values when choosing the `multistep` scheduler. Default to `20`)

[//]: # ()
[//]: # (- `gamma`: Learning rate ratio for `step` or `multistep` scheduler, default to `0.2`)

[//]: # ()
[//]: # (- `fix_BN`: Set the encoder to the evaluation mode during the meta-training. This parameter is useful when meta-learning with the WRN. Default to `False`)

[//]: # ()
[//]: # (- `augment`: Whether to do data augmentation or not during meta-training, default to `False`)

[//]: # ()
[//]: # (- `mom`: The momentum value for the SGD optimizer, default to `0.9`)

[//]: # ()
[//]: # (- `weight_decay`: The weight_decay value for SGD optimizer, default to `0.0005`)

[//]: # ()
[//]: # (**Model Related Arguments**)

[//]: # (- `model_class`: The model to use during meta-learning. We provide implementations for baselines &#40;`MatchNet` and `ProtoNet`&#41;, set-to-set functions &#40;`BILSTM`, `DeepSet`, `GCN`, and our `FEAT` variants&#41;. We also include an instance-specific embedding adaptation approach `FEAT`, which is discussed in the old version of the paper. `SemiFEAT` is the one which combines the unlabeled query set instances into the feature adaptation in a transductive manner, while `SemiProtoFEAT` applies Semi-ProtoNet over the transductively transformed embeddings of `SemiFEAT`. Default to `FEAT`)

[//]: # ()
[//]: # (- `use_euclidean`: Use the euclidean distance or the cosine similarity to compute pairwise distances. We use the euclidean distance in the paper. Default to `False`)

[//]: # ()
[//]: # (- `backbone_class`: Types of the encoder, i.e., the convolution network &#40;`ConvNet`&#41;, ResNet-12 &#40;`Res12`&#41;, or Wide ResNet &#40;`WRN`&#41;, default to `ConvNet`)

[//]: # ()
[//]: # (- `balance`: This is the balance weight for the contrastive regularizer. Default to `0`)

[//]: # ()
[//]: # (- `temperature`: Temperature over the logits, we #divide# logits with this value. It is useful when meta-learning with pre-trained weights. Default to `1`)

[//]: # ()
[//]: # (- `temperature2`: Temperature over the logits in the regularizer, we divide logits with this value. This is specially designed for the contrastive regularizer. Default to `1`)

[//]: # ()
[//]: # (**Other Arguments** )

[//]: # (- `orig_imsize`: Whether to resize the images before loading the data into the memory. `-1` means we do not resize the images and do not read all images into the memory. Default to `-1`)

[//]: # ()
[//]: # (- `multi_gpu`: Whether to use multiple gpus during meta-training, default to `False`)

[//]: # ()
[//]: # (- `gpu`: The index of GPU to use. Please provide multiple indexes if choose `multi_gpu`. Default to `0`)

[//]: # ()
[//]: # (- `log_interval`: How often to log the meta-training information, default to every `50` tasks)

[//]: # ()
[//]: # (- `eval_interval`: How often to validate the model over the meta-val set, default to every `1` epoch)

[//]: # ()
[//]: # (- `save_dir`: The path to save the learned models, default to `./checkpoints`)

[//]: # ()
[//]: # (Running the command without arguments will train the models with the default hyper-parameter values. Loss changes will be recorded as a tensorboard file.)

[//]: # ()
[//]: # (## Training scripts for FEAT)

[//]: # ()
[//]: # (For example, to train the 1-shot/5-shot 5-way FEAT model with ConvNet backbone on MiniImageNet:)

[//]: # ()
[//]: # (    $ python train_fsl.py  --max_epoch 200 --model_class FEAT --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 1 --temperature 64 --temperature2 16 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 8 --init_weights ./saves/initialization/miniimagenet/con-pre.pth --eval_interval 1)

[//]: # (    $ python train_fsl.py  --max_epoch 200 --model_class FEAT --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 64 --lr 0.0001 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 14 --init_weights ./saves/initialization/miniimagenet/con-pre.pth --eval_interval 1)

[//]: # ()
[//]: # (to train the 1-shot/5-shot 5-way FEAT model with ResNet-12 backbone on MiniImageNet:)

[//]: # ()
[//]: # (    $ python train_fsl.py  --max_epoch 200 --model_class FEAT  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 1 --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean)

[//]: # (    $ python train_fsl.py  --max_epoch 200 --model_class FEAT  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 64 --temperature2 32 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 0 --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean)

[//]: # ()
[//]: # (to train the 1-shot/5-shot 5-way FEAT model with ResNet-12 backbone on TieredImageNet:)

[//]: # ()
[//]: # (    $ python train_fsl.py  --max_epoch 200 --model_class FEAT  --backbone_class Res12 --dataset TieredImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.1 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu 0 --init_weights ./saves/initialization/tieredimagenet/Res12-pre.pth --eval_interval 1  --use_euclidean)

[//]: # (    $ python train_fsl.py  --max_epoch 200 --model_class FEAT  --backbone_class Res12 --dataset TieredImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu 0 --init_weights ./saves/initialization/tieredimagenet/Res12-pre.pth --eval_interval 1  --use_euclidean)

[//]: # ()
[//]: # (## Acknowledgment)

[//]: # (We thank the following repos providing helpful components/functions in our work.)

[//]: # (- [ProtoNet]&#40;https://github.com/cyvius96/prototypical-network-pytorch&#41;)

[//]: # ()
[//]: # (- [MatchingNet]&#40;https://github.com/gitabcworld/MatchingNetworks&#41;)

[//]: # ()
[//]: # (- [PFA]&#40;https://github.com/joe-siyuan-qiao/FewShot-CVPR/&#41;)

[//]: # ()
[//]: # (- [Transformer]&#40;https://github.com/jadore801120/attention-is-all-you-need-pytorch&#41;)

[//]: # ()
[//]: # (- [MetaOptNet]&#40;https://github.com/kjunelee/MetaOptNet/&#41;)

