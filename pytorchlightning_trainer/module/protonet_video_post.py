from pathlib import Path

import torch
from torch.nn import functional as F
import torchmetrics

from src.official_orbit.utils.data import attach_frame_history
from src.utils.evaluator import FEATEpisodeEvaluator
from src.data.orbit_few_shot_video_classification import \
    get_object_category_names_according_labels
from pytorchlightning_trainer.module.protonet import ProtoNetWithLITE
from src.utils.missing_object_detector import filter_support_clips_without_object


class ProtoNetWithLITEVideoPost(ProtoNetWithLITE):
    """
        This class implements ProtoNet using LITE, also integrated with the video support clips selection
        using Canny Edge Detector during testing.

    """
    def on_test_start(self) -> None:
        self.model._set_device(self.device)
        self.model.set_test_mode(True)
        self.episode_evaluator = FEATEpisodeEvaluator(
            save_dir=str(Path(self.trainer.logger.root_dir, "testing_per_video_results")))

    def test_step(self, val_batch, batch_idx):
        support_clips_frames = val_batch['support_frames']
        support_clips_labels = val_batch['support_labels']
        support_clips_filenames = val_batch['support_frame_filenames']

        object_category_names = \
            get_object_category_names_according_labels(val_batch['query_frame_filenames'], val_batch['query_labels'])
        self.episode_evaluator.register_object_category(object_category_names)

        # Filter non-object-present support clips: apply the edge detector to identify which frame has
        # non-object-present issue, and remove the related support clips
        if "raw_support_frames" not in val_batch:
            raise ValueError("The edge detector must take inputs raw RGB frames, but not included in the dataloader.")
        num_total_support_clips = len(support_clips_frames)
        support_clips_frames, support_clips_labels, support_clips_filenames = \
            filter_support_clips_without_object(val_batch["raw_support_frames"], support_clips_frames, support_clips_labels, support_clips_filenames)
        num_valid_support_clips = len(support_clips_frames)
        print("In user: {}, num_valid_support_clips / total_num_support_clips = {} / {} = {}".
              format(val_batch["user_id"], num_valid_support_clips, num_total_support_clips, num_valid_support_clips / num_total_support_clips))

        self.model.personalise(support_clips_frames, support_clips_labels)

        # Register sampled support clips and their features for post-analysis (Optional)
        if self.register_testing_supports:
            prototypes = torch.cat(list(self.model.classifier.adapted_class_rep_dict.values()), dim=0)
            if len(prototypes) != len(object_category_names):
                raise ValueError("In the current episode, the number of prototypes "
                                 "is not equal to the number of object categories")
            self.episode_evaluator.register_prototypes(prototypes.cpu().numpy())
            self.episode_evaluator.add_multiple_clips_results(clips_features=self.model.context_features.cpu().numpy(),
                                                              clips_filenames=support_clips_filenames,
                                                              clips_labels=support_clips_labels.cpu().numpy())

        for video_sequence_frames,  video_sequence_label, video_frame_filenames in \
                zip(val_batch['query_frames'], val_batch['query_labels'],val_batch['query_frame_filenames']):
            video_clips_frames = attach_frame_history(video_sequence_frames, self.video_clip_length)
            video_logits, video_features = self.model.predict(video_clips_frames)  # [num_frames, num_classes]
            video_prediction_scores = F.softmax(video_logits, dim=-1)
            video_predictions = video_logits.argmax(dim=-1).cpu()
            num_frames = video_logits.shape[0]
            video_labels = video_sequence_label.expand(num_frames).cpu()
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

