import os.path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from pytorchmetalearning.data.utils import sort_dictionary_with_key
from pytorchmetalearning.data.datasets.orbit_few_shot_video_classification import \
    recover_video_frames_folder_fullpath_from_video_name



def load_episode_result(filename: str):
    print("Load {}".format(filename))
    with open(filename, "rb") as f:
        episode_result = pickle.load(f)
    return episode_result


def generate_per_frame_database(episode_result):
    """
        # filename, prediction, score, gt_label, is_prototype
    """
    database_components = []
    # Step 1: Add support results
    temp_dict = {"filename_fullpath": [],
                 "filename": [],
                 "prediction": [],
                 "score": [],
                 "gt_label": [],
                 "support, query, prototype": [],
                 "marker_size": []}
    for clip_result in episode_result.support_clips_results:
        temp_dict["filename_fullpath"].append(clip_result.per_frame_filenames[0])
        temp_dict["filename"].append(str(Path(clip_result.per_frame_filenames[0]).name))
        temp_dict["gt_label"].append(clip_result.gt_label)
        temp_dict["support, query, prototype"].append("support")
        temp_dict["marker_size"].append(1.5)
    support_database = pd.DataFrame.from_dict(temp_dict, orient='index').T
    database_components.append(support_database)

    # Step 2: Add query results
    temp_dict = {"filename_fullpath": [],
                 "filename": [],
                 "prediction": [],
                 "score": [],
                 "gt_label": [],
                 "support, query, prototype": [],
                 "marker_size": []}
    for video_result in episode_result.query_video_results:
        temp_dict["filename_fullpath"].extend(video_result.per_frame_filenames)
        temp_dict["filename"].extend([str(Path(filename).name) for filename in video_result.per_frame_filenames])
        temp_dict["prediction"].extend(video_result.per_frame_predictions)
        temp_dict["score"].extend(video_result.per_frame_scores)
        num_frames = len(video_result.per_frame_filenames)
        temp_dict["gt_label"].extend([video_result.gt_label] * num_frames)
        temp_dict["support, query, prototype"].extend(["query"] * num_frames)
        temp_dict["marker_size"].extend([0.5] * num_frames)
    query_database = pd.DataFrame.from_dict(temp_dict)
    database_components.append(query_database)

    # Step 3: Add prototypes
    num_prototypes = len(episode_result.prototypes)
    temp_dict = {"filename": [],
                 "filename_fullpath": [],
                 "prediction": [],
                 "score": [],
                 "gt_label": list(range(num_prototypes)),
                 "support, query, prototype": ["proto"] * num_prototypes,
                 "marker_size": [4.1] * num_prototypes}
    prototype_database = pd.DataFrame.from_dict(temp_dict, orient='index').T
    database_components.append(prototype_database)

    # Step 4: Add original prototypes
    if hasattr(episode_result, "original_prototypes"):
        num_prototypes = len(episode_result.original_prototypes)
        temp_dict = {"filename": [],
                     "filename_fullpath": [],
                     "prediction": [],
                     "score": [],
                     "gt_label": list(range(num_prototypes)),
                     "support, query, prototype": ["original_proto"] * num_prototypes,
                     "marker_size": [4.1] * num_prototypes}
        original_prototype_database = pd.DataFrame.from_dict(temp_dict, orient='index').T
        database_components.append(original_prototype_database)

    database = pd.concat(database_components)
    # database = pd.concat([support_database, prototype_database])
    return database


def generate_tsne_outputs(episode_result):
    """
        features: [1, num_channel], default=1280
    """
    feature_maps = []
    # Add support results
    for clip_result in episode_result.support_clips_results:
        feature_maps.append(clip_result.per_clip_feature)

    # Add query results
    for video_result in episode_result.query_video_results:
        feature_maps.extend(video_result.per_frame_features)

    # Add prototype results
    feature_maps.extend(episode_result.prototypes)

    # Add original prototype results
    if hasattr(episode_result, "original_prototypes"):
        feature_maps.extend(episode_result.original_prototypes)

    # Calculate TSNE
    feature_maps = np.concatenate([feature.reshape(1, 1280) for feature in feature_maps], axis=0)
    print("Apply tTNE on {} samples...".format(len(feature_maps)))
    tsne = TSNE(random_state=0,
                n_jobs=2,
                n_components=2,
                learning_rate='auto',
                init='random',
                # n_iter= 250
                )
    tsne_output = tsne.fit_transform(feature_maps)
    print("Done...")
    return tsne_output


def generate_tsne_outputs_methods(episode_result1, episode_result2):
    """
        features: [1, num_channel], default=1280
    """
    feature_maps = []
    # Add support results
    for clip_result in episode_result1.support_clips_results:
        feature_maps.append(clip_result.per_clip_feature)

    # Add query results
    for video_result in episode_result1.query_video_results:
        feature_maps.extend(video_result.per_frame_features)

    # Add prototype results
    feature_maps.extend(episode_result1.prototypes)

    # Add original prototype results
    if hasattr(episode_result1, "original_prototypes"):
        feature_maps.extend(episode_result1.original_prototypes)

    # 2
    # Add support results
    for clip_result in episode_result2.support_clips_results:
        feature_maps.append(clip_result.per_clip_feature)

    # Add query results
    for video_result in episode_result2.query_video_results:
        feature_maps.extend(video_result.per_frame_features)

    # Add prototype results
    feature_maps.extend(episode_result2.prototypes)

    # Add original prototype results
    if hasattr(episode_result2, "original_prototypes"):
        feature_maps.extend(episode_result2.original_prototypes)

    # Calculate TSNE
    feature_maps = np.concatenate([feature.reshape(1, 1280) for feature in feature_maps], axis=0)
    print("Apply tTNE on {} samples...".format(len(feature_maps)))
    tsne = TSNE(random_state=0,
                n_jobs=2,
                n_components=2,
                learning_rate='auto',
                init='random',
                # n_iter= 250
                )
    tsne_output = tsne.fit_transform(feature_maps)
    print("Done...")
    return tsne_output


def get_frame_info(video_name: str, frame_index: int, subset: str = "train"):
    video_frames_folder = recover_video_frames_folder_fullpath_from_video_name(video_name, subset)
    frame_filename = "-".join([video_name, str(frame_index).zfill(5) + ".jpg"])
    frame_fullpath = os.path.join(video_frames_folder, frame_filename)
    if not os.path.exists(frame_fullpath):
        raise ValueError("{} does not exist !".format(frame_fullpath))
    return frame_fullpath


def prepare_dataframe(filename):
    episode_result = load_episode_result(filename)
    mapping = {label: name for label, name in enumerate(episode_result.object_names)}
    print(mapping)
    database = generate_per_frame_database(episode_result)
    tsne_output = generate_tsne_outputs(episode_result)
    database["x"] = tsne_output[:, 0]
    database["y"] = tsne_output[:, 1]
    database = database.astype({"marker_size": 'float'})
    # sns.scatterplot(data=database, x="x", y="y", hue="gt_label", style="is_prototype", palette="deep", )
    # plt.show()

    total_acc = []
    print("-" * 50)
    print("Each video's frame_accuracy:")
    for video_result in episode_result.query_video_results:
        print("{}= {}".format(video_result.video_name, video_result.video_frame_accuracy))
        total_acc.append(video_result.video_frame_accuracy)
    print("-" * 50)
    print("Average acc of {} = {}".format(episode_result.query_video_results[0].user_id, sum(total_acc) / len(total_acc)))
    return database


def combine_methods_tsne(exp_name1, exp_name_2):
    episode_result_1 = load_episode_result(exp_name1)
    database_1 = generate_per_frame_database(episode_result_1)
    method1 = ["protonet_original"] * len(database_1)
    database_1["method"] = method1

    episode_result_2 = load_episode_result(exp_name_2)
    database_2 = generate_per_frame_database(episode_result_2)
    method2 = ["ours"] * len(database_2)
    database_2["method"] = method2

    c = pd.concat([database_1, database_2])

    tsne_output = generate_tsne_outputs_methods(episode_result_1, episode_result_2)
    c["x"] = tsne_output[:, 0]
    c["y"] = tsne_output[:, 1]
    c = c.astype({"marker_size": 'float'})
    return c





#
# def draw_tsne(mapping_label_to_video_features, mapping_label_to_prototype):
#     if mapping_label_to_video_features.keys() != mapping_label_to_prototype.keys():
#         raise ValueError("Error! the query frames and the support frames come from different object categories")
#     if not mapping_label_to_prototype:
#         raise ValueError("Please register prototypes from the support set at first !")
#
#     print('generating t-SNE plot...')
#     mapping_label_to_video_features = sort_dictionary_with_key(mapping_label_to_video_features)
#
#     indices = [features.shape[0] for _, features in mapping_label_to_video_features.items()]
#     indices.append(len(list(mapping_label_to_prototype.keys())))
#
#     outputs = torch.cat(
#         [features for _, features in mapping_label_to_video_features.items()] + [prototype for _, prototype in
#                                                                                  mapping_label_to_prototype.items()],
#         dim=0)
#     targets = torch.cat(
#         [torch.ones(len(features)) * label for label, features in mapping_label_to_video_features.items()] + [
#             torch.tensor(label).unsqueeze(dim=0) for label in list(mapping_label_to_prototype.keys())], dim=0)
#     targets = targets.numpy().astype(int)
#
#     tsne = TSNE(random_state=0)
#     tsne_output = tsne.fit_transform(outputs)
#
#     # x, y, label, prototype
#     df = pd.DataFrame(tsne_output, columns=['x', 'y'])
#     df["label"] = targets
#     prototype_flag = np.zeros_like(targets)
#     prototype_flag[-indices[-1]:] = 1
#     df["prototype"] = prototype_flag
#     sns.scatterplot(data=df, x="x", y="y", hue="label", style="prototype", palette="deep", )
#     # #
#     # # plt.rcParams['figure.figsize'] = 5, 5
#     # title = "/".join([name for _, name in self.mapping_label_to_object_category.items()])
#     # plt.title(title)
#     #
#     # filename = "_".join([name for _, name in self.mapping_label_to_object_category.items()]) + ".png"
#     # plt.savefig(os.path.join(self.save_dir, filename), bbox_inches='tight')
#     # plt.clf()
#     #
#     # plt.savefig(os.path.join(save_dir, 'tsne.png'), bbox_inches='tight')
#     # print('done!')


if __name__ == '__main__':
    filename = "/home/ligu/projects/orbit_challenge_2022_refactor/logs/tb_logs/orbit_protonet_baseline/testing_per_video_results/P953.p"
    # prepare_dataframe(filename)

    exp_name1 = "/home/ligu/projects/orbit_challenge_2022_refactor/logs/tb_logs/orbit_protonet_baseline/testing_per_video_results/P953.p"
    exp_name_2 = "/home/ligu/projects/orbit_challenge_2022_refactor/logs/tb_logs/test_best_data_aug_uniform_chunk10_video_post_only_support_1_thres0.0005_reproduce_our_submission/testing_per_video_results/P953.p"
    combine_methods_tsne(exp_name1, exp_name_2)
