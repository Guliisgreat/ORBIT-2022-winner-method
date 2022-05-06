from typing import Optional
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision

import hydra
import omegaconf

from omegaconf import DictConfig
from pytorchlightning_trainer.utils.common import PROJECT_ROOT

from pytorch_lightning import LightningDataModule
from pytorch_lightning import seed_everything

from src.transforms import ApplyTransformToKey, Div255, MutuallyExclusiveLabel
from src.data import ORBITUserCentricVideoFewShotClassification


def custom_collate_fn(batch):
    """
        Select the 1st episode from each meta batch
        TODO: Use distributed data parallel to process multiple episodes in each meta batch
    """
    return batch[0]


class ORBITDataModule(LightningDataModule):
    """
        Arguments
            Three Splits: train, val, test
            Default config file: pytorchlightning_trainer/conf/data/default.yaml
    """

    def __init__(self,
                 root: str = "/data02/datasets/ORBIT_microsoft",
                 use_orbit_statistics: bool = False,
                 train_cfg: DictConfig = None,
                 val_cfg: DictConfig = None,
                 test_cfg: DictConfig = None,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.root = root
        self.use_orbit_statistics = use_orbit_statistics
        if not isinstance(train_cfg, DictConfig):
            raise ValueError("Lack of the config file for training data pipeline")
        if not isinstance(val_cfg, DictConfig):
            raise ValueError("Lack of the config file for validation data pipeline")
        if not isinstance(test_cfg, DictConfig):
            raise ValueError("Lack of the config file for testing data pipeline")
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        self.test_cfg = test_cfg

    def setup(self, stage: Optional[str] = None):
        if self.use_orbit_statistics:
            stats = torchvision.transforms.Normalize(mean=(0.500, 0.436, 0.396), std=(0.145, 0.143, 0.138))
        else:
            stats = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        transform = torchvision.transforms.Compose([
            ApplyTransformToKey(
                key="support_image",
                transform=torchvision.transforms.Compose([
                    Div255(),
                    stats,
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
                transform=transform,
                num_threads=self.train_cfg.num_threads,
                use_object_cluster_labels=self.train_cfg.use_object_cluster_labels,
                mode="train")

        self.test_datasets = \
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
                transform=transform,
                num_threads=self.val_cfg.num_threads,
                use_object_cluster_labels=self.val_cfg.use_object_cluster_labels,
                mode="test")

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
                transform=transform,
                num_threads=self.test_cfg.num_threads,
                use_object_cluster_labels=self.test_cfg.use_object_cluster_labels,
                mode="test")

    # return the dataloader for each split
    def train_dataloader(self):
        return DataLoader(dataset=self.train_datasets,
                          batch_size=1,
                          shuffle=True,
                          num_workers=self.train_cfg.num_workers,
                          prefetch_factor=2,
                          collate_fn=custom_collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_datasets,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.val_cfg.num_workers,
                          collate_fn=custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_datasets,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.test_cfg.num_workers,
                          collate_fn=custom_collate_fn)


@hydra.main(config_path=str(PROJECT_ROOT / "pytorchlightning_trainer/conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    seed_everything(42)

    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup()
    for idx, batch in enumerate(tqdm(datamodule.test_dataloader())):
        if idx == 30:
            break


if __name__ == '__main__':
    main()