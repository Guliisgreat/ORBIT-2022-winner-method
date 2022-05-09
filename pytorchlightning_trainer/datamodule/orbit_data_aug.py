from typing import Optional

import torchvision
from src.transforms import ApplyTransformToKey, Div255, MutuallyExclusiveLabel
from src.data import ORBITUserCentricVideoFewShotClassification
from pytorchlightning_trainer.datamodule import ORBITDataModule


class ORBITDataModuleDataAug(ORBITDataModule):
    """
        This datamodule class implements the ORBIT data pipeline using several common data augmentation techniques
        to increase the diversity of episodes during training
    """
    def setup(self, stage: Optional[str] = None):
        if self.use_orbit_statistics:
            stats = torchvision.transforms.Normalize(mean=(0.500, 0.436, 0.396), std=(0.145, 0.143, 0.138))
        else:
            stats = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        train_transform = torchvision.transforms.Compose([
            ApplyTransformToKey(
                key="support_image",
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.RandomVerticalFlip(p=0.5),
                    torchvision.transforms.RandomRotation(90),
                    # torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                    Div255(),
                    stats,
                ])
            ),
            ApplyTransformToKey(
                key="query_image",
                transform=torchvision.transforms.Compose([
                    # torchvision.transforms.Resize(84),
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    torchvision.transforms.RandomVerticalFlip(p=0.5),
                    torchvision.transforms.RandomRotation(90),
                    # torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                    Div255(),
                    stats,
                ])
            ),
            ApplyTransformToKey(
                key="episode_label",
                transform=torchvision.transforms.Compose([
                    MutuallyExclusiveLabel(shuffle_ordered_label=True),
                ])
            ),
        ])

        test_transform = torchvision.transforms.Compose([
            ApplyTransformToKey(
                key="support_image",
                transform=torchvision.transforms.Compose([
                    Div255(),
                    stats
                ])
            ),
            ApplyTransformToKey(
                key="query_image",
                transform=torchvision.transforms.Compose([
                    Div255(),
                    stats
                ])
            ),
            ApplyTransformToKey(
                key="episode_label",
                transform=torchvision.transforms.Compose([
                    MutuallyExclusiveLabel(shuffle_ordered_label=True),
                ])
            ),
        ])

        self.train_datasets = \
            ORBITUserCentricVideoFewShotClassification(
                data_path=self.train_cfg.root,
                object_sampler=self.train_cfg.object_sampler,
                max_num_objects_per_user=self.train_cfg.max_num_objects_per_user,
                support_instance_sampler=self.train_cfg.support_instance_sampler,
                query_instance_sampler=self.train_cfg.query_instance_sampler,
                support_num_shot=self.train_cfg.support_num_shot,
                query_num_shot=self.train_cfg.query_num_shot,
                query_video_type=self.train_cfg.query_video_type,
                max_num_instance_per_object=self.train_cfg.max_num_instance_per_object,
                support_clip_sampler=self.train_cfg.support_clip_sampler,
                query_clip_sampler=self.train_cfg.query_clip_sampler,
                video_subsample_factor=self.train_cfg.video_subsample_factor,
                num_episodes_per_user=self.train_cfg.num_episodes_per_user,
                video_clip_length=self.train_cfg.video_clip_length,
                max_num_clips_per_video=self.train_cfg.max_num_clips_per_video,
                transform=train_transform,
                num_threads=self.train_cfg.num_threads,
                use_object_cluster_labels=self.train_cfg.use_object_cluster_labels,
                mode="train")

        self.val_datasets = \
            ORBITUserCentricVideoFewShotClassification(
                data_path=self.val_cfg.root,
                object_sampler=self.val_cfg.object_sampler,
                max_num_objects_per_user=self.val_cfg.max_num_objects_per_user,
                support_instance_sampler=self.val_cfg.support_instance_sampler,
                query_instance_sampler=self.val_cfg.query_instance_sampler,
                query_video_type=self.val_cfg.query_video_type,
                max_num_instance_per_object=self.val_cfg.max_num_instance_per_object,
                support_clip_sampler=self.val_cfg.support_clip_sampler,
                query_clip_sampler=self.val_cfg.query_clip_sampler,
                video_subsample_factor=self.val_cfg.video_subsample_factor,
                num_episodes_per_user=self.val_cfg.num_episodes_per_user,
                video_clip_length=self.val_cfg.video_clip_length,
                max_num_clips_per_video=self.val_cfg.max_num_clips_per_video,
                transform=test_transform,
                num_threads=self.val_cfg.num_threads,
                use_object_cluster_labels=self.val_cfg.use_object_cluster_labels,
                mode="validation")

        self.test_datasets = \
            ORBITUserCentricVideoFewShotClassification(
                data_path=self.test_cfg.root,
                object_sampler=self.test_cfg.object_sampler,
                max_num_objects_per_user=self.test_cfg.max_num_objects_per_user,
                support_instance_sampler=self.test_cfg.support_instance_sampler,
                query_instance_sampler=self.test_cfg.query_instance_sampler,
                query_video_type=self.test_cfg.query_video_type,
                max_num_instance_per_object=self.test_cfg.max_num_instance_per_object,
                support_clip_sampler=self.test_cfg.support_clip_sampler,
                query_clip_sampler=self.test_cfg.query_clip_sampler,
                video_subsample_factor=self.test_cfg.video_subsample_factor,
                num_episodes_per_user=self.test_cfg.num_episodes_per_user,
                video_clip_length=self.test_cfg.video_clip_length,
                max_num_clips_per_video=self.test_cfg.max_num_clips_per_video,
                transform=test_transform,
                num_threads=self.test_cfg.num_threads,
                use_object_cluster_labels=self.test_cfg.use_object_cluster_labels,
                mode="test")
