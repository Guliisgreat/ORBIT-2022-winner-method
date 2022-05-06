from pathlib import Path

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics

from src.official_orbit.models.few_shot_recognisers import SingleStepFewShotRecogniser
from src.official_orbit.utils.data import get_clip_loader, attach_frame_history
from src.utils.evaluator import EpisodeEvaluator, FEATEpisodeEvaluator, convert_results_in_submission_format, \
    compute_average_frame_accuracy_across_videos, calculate_frame_accuracy
from src.data.orbit_few_shot_video_classification import \
    get_object_category_names_according_labels


class ProtoNetWithLITE(pl.LightningModule):
    def __init__(self,
                 pretrained_backbone_checkpoint_path: str,
                 backbone_network: str = "efficientnetb0",
                 unfreeze_backbone: bool = True,
                 freeze_BN_layer: bool = True,
                 normalization_layer: str = "basic",
                 use_adapt_features: bool = False,
                 feature_adaptation_method: str = "generate",
                 classifier: str = "proto",
                 video_clip_length: int = 8,
                 episode_subset_mini_batch_size: int = 8,
                 num_episodes_per_meta_batch: int = 16,
                 num_lite_samples: int = 8,
                 lr: float = 0.0001,
                 lr_scheduler_step_milestone: float = 0.7,
                 lr_scheduler_gamma: float = 0.5,
                 use_two_gpus: bool = False,
                 register_testing_supports: bool = True,
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.model = SingleStepFewShotRecogniser(
            pretrained_extractor_path=pretrained_backbone_checkpoint_path,
            feature_extractor=backbone_network,
            batch_normalisation=normalization_layer,
            adapt_features=use_adapt_features,
            classifier=classifier,
            clip_length=video_clip_length,
            batch_size=episode_subset_mini_batch_size,
            learn_extractor=unfreeze_backbone,
            feature_adaptation_method=feature_adaptation_method,
            num_lite_samples=num_lite_samples,
            use_two_gpus=use_two_gpus,
        )
        self.model._register_extra_parameters()
        self.model.set_test_mode(False)

        # TODO: Try other normalization layers
        if normalization_layer != "basic":
            raise NotImplementedError("Please use Batch Norm in ProtoNet in default")
        self.pretrained_backbone_checkpoint_path = pretrained_backbone_checkpoint_path
        self.freeze_BN_layer = freeze_BN_layer
        self.num_episodes_per_meta_batch = num_episodes_per_meta_batch
        self.episode_subset_mini_batch_size = episode_subset_mini_batch_size
        self.num_lite_examples = num_lite_samples
        self.use_two_gpus = use_two_gpus
        self.video_clip_length = video_clip_length
        self.lr = lr
        self.lr_scheduler_step_milestone = lr_scheduler_step_milestone,
        self.lr_scheduler_gamma = lr_scheduler_gamma,
        self.register_testing_supports = register_testing_supports

        # TODO: Freeze BN layer??
        if self.freeze_BN_layer:
            self.model.set_test_mode(True)
            print("BN layers are frozen. Its parameters and statistics will be kept same with those from ImageNet ")
        # Initialize evaluators (frame accuracy)
        self.train_metrics = torchmetrics.Accuracy()
        # self.val_metrics = torchmetrics.Accuracy()
        # unify evaluation calculations
        self.val_metrics = []
        self.episode_evaluator = None

        self.automatic_optimization = False

    def on_train_start(self) -> None:
        # Hard code; To avoid direct modifications on official_orbit.models.few_shot_recognizers.py
        self.model._set_device(self.device)

    def training_step(self, train_batch, batch_idx):

        support_multi_clips_frames = train_batch['support_frames']
        support_multi_clips_labels = train_batch['support_targets']
        query_multi_clips_frames = train_batch['query_frames']
        query_multi_clips_labels = train_batch['query_targets']

        episode_total_loss = 0

        self.model._cache_context_outputs(support_multi_clips_frames)
        num_support_clips = support_multi_clips_frames.shape[0]
        query_batched_clips_loader = get_clip_loader((query_multi_clips_frames, query_multi_clips_labels),
                                                     self.episode_subset_mini_batch_size, with_labels=True)
        # # Remove the last mini_batch, because its number of clips < episode_subset_mini_batch_size
        # query_batched_clips_loader.pop(-1)

        for min_batched_query_clips_frames, mini_batched_query_labels in query_batched_clips_loader:
            self.model.personalise_with_lite(support_multi_clips_frames,
                                             support_multi_clips_labels)  # 73 support clips; 8 LITE backward
            min_batch_query_prediction_logits = self.model.predict_a_batch(min_batched_query_clips_frames)
            loss_scaling = num_support_clips / (self.num_lite_examples * self.num_episodes_per_meta_batch)
            batch_loss = loss_scaling * F.cross_entropy(min_batch_query_prediction_logits, mini_batched_query_labels,
                                                        reduction="mean")
            batch_loss += 0.001 * self.model.feature_adapter.regularization_term(switch_device=self.use_two_gpus)
            episode_total_loss += batch_loss.detach()
            self.train_metrics.update(min_batch_query_prediction_logits, mini_batched_query_labels)

            self.manual_backward(batch_loss)
            # reset task's params
            self.model.classifier.reset()

        self.log("train_loss_per_episode", episode_total_loss / len(query_batched_clips_loader), on_step=True,
                 on_epoch=False)

        self.lr_schedulers().step()

        # accumulate gradients of each episode, and compute the average meta gradient of the meta batch
        # num of episodes in each meta batch = 16 (default)
        if (batch_idx + 1) % self.num_episodes_per_meta_batch == 0:
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

            average_meta_batch_acc = self.train_metrics.compute()
            self.log("train_accuracy_average_per_meta_batch", average_meta_batch_acc, on_step=True, on_epoch=False)
            self.train_metrics.reset()

    def on_validation_start(self) -> None:
        # Hard code; To avoid direct modifications on official_orbit.models.few_shot_recognizers.py
        self.model._set_device(self.device)

    def validation_step(self, val_batch, batch_idx):

        support_clips_frames = val_batch['support_frames']
        support_clips_labels = val_batch['support_targets']

        self.model.personalise(support_clips_frames, support_clips_labels)

        # loop through cached target videos for the current task
        total_loss = 0
        for video_sequence_frames, video_sequence_label in zip(val_batch['query_frames'], val_batch['query_targets']):
            video_clips_frames = attach_frame_history(video_sequence_frames, self.video_clip_length)
            video_logits, _ = self.model.predict(video_clips_frames)  # [num_frames, num_classes]
            num_frames = video_logits.shape[0]

            video_predictions = video_logits.argmax(dim=-1)
            self.val_metrics.append(
                calculate_frame_accuracy(video_sequence_label.item(), video_predictions.cpu().tolist()))

            video_labels = video_sequence_label.expand(num_frames).to(device=video_logits.device)
            total_loss += F.cross_entropy(video_logits, video_labels, reduction="mean")
            # self.val_metrics.update(video_logits, video_labels)

        self.model._reset()
        self.log("val_loss", total_loss / len(val_batch['query_targets']), on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self.model.set_test_mode(True)

    def on_validation_epoch_end(self) -> None:
        avg_val_acc = sum(self.val_metrics) / len(self.val_metrics)
        # self.log("val_acc", self.val_metrics.compute())
        self.log("val_acc", avg_val_acc)
        # self.val_metrics.reset()
        self.val_metrics = []

        if not self.freeze_BN_layer:
            self.model.set_test_mode(False)

    def on_test_start(self) -> None:
        # Hard code; To avoid direct modifications on official_orbit.models.few_shot_recognizers.py
        self.model._set_device(self.device)
        self.model.set_test_mode(True)
        self.episode_evaluator = EpisodeEvaluator(
            save_dir=str(Path(self.trainer.logger.root_dir, "testing_per_video_results")))

    def test_step(self, val_batch, batch_idx):

        support_clips_frames = val_batch['support_frames']
        support_clips_labels = val_batch['support_targets']
        support_clips_filenames = val_batch['support_frame_filenames']

        object_category_names = \
            get_object_category_names_according_labels(val_batch['query_frame_filenames'], val_batch['query_targets'])
        self.episode_evaluator.register_object_category(object_category_names)

        self.model.personalise(support_clips_frames, support_clips_labels)

        # Register sampled support clips and their features for post-analysis (Optional)
        if self.register_testing_supports:
            prototypes = self.model.classifier.param_dict['weight']  # [num_object, 1280]
            if len(prototypes) != len(object_category_names):
                raise ValueError("In the current episode, the number of prototypes "
                                 "is not equal to the number of object categories")
            self.episode_evaluator.register_prototypes(prototypes.cpu().numpy())
            self.episode_evaluator.add_multiple_clips_results(clips_features=self.model.context_features.cpu().numpy(),
                                                              clips_filenames=support_clips_filenames,
                                                              clips_labels=support_clips_labels.cpu().numpy())

        for video_sequence_frames, video_sequence_label, video_frame_filenames in \
                zip(val_batch['query_frames'], val_batch['query_targets'], val_batch['query_frame_filenames']):
            video_clips_frames = attach_frame_history(video_sequence_frames, self.video_clip_length)
            video_logits, video_features = self.model.predict(video_clips_frames)  # [num_frames, num_classes]
            video_prediction_scores = F.softmax(video_logits, dim=-1)
            video_predictions = video_logits.argmax(dim=-1)
            num_frames = video_logits.shape[0]
            video_labels = video_sequence_label.expand(num_frames).to(device=video_logits.device)
            acc = torchmetrics.functional.accuracy(video_predictions, video_labels)
            self.episode_evaluator.add_video_result(per_frame_prediction_scores=video_prediction_scores.cpu().numpy(),
                                                    per_frame_features=video_features.cpu().numpy(),
                                                    frame_filenames=video_frame_filenames,
                                                    video_gt_label=video_sequence_label.item(),
                                                    video_frame_accuracy=acc.item())
        self.model._reset()

        self.episode_evaluator.compute_statistics()
        self.episode_evaluator.save_to_disk()
        self.episode_evaluator.reset()

    def on_test_end(self) -> None:
        convert_results_in_submission_format(self.episode_evaluator.save_dir)
        compute_average_frame_accuracy_across_videos(self.episode_evaluator.save_dir)

    def configure_optimizers(self):
        feature_extractor_params = list(map(id, self.model.feature_extractor.parameters()))
        base_params = filter(lambda p: id(p) not in feature_extractor_params, self.model.parameters())
        # TODO: Replace Adam with SGD
        # optimizer_fn = optimizers[optimizer_type]
        extractor_scale_factor = 0.1 if self.pretrained_backbone_checkpoint_path else 1.0
        # optimizer = torch.optim.SGD([
        #     {'params': base_params},
        #     {'params': self.model.feature_extractor.parameters(), 'lr': self.lr * extractor_scale_factor}
        # ], lr=self.lr, momentum=0.9)

        optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': self.model.feature_extractor.parameters(), 'lr': self.lr * extractor_scale_factor}
        ], lr=self.lr)
        print("initial learning rate = {}".format(self.lr))

        total_num_train_episodes = len(self.trainer.datamodule.train_dataloader())
        milestone = int(0.7 * 22000)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone, ],
                                                                  gamma=0.5),
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        # return optimizer
