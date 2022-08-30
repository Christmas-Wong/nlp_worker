# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 14:06
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : util.py
@Software   : PyCharm
@Description: 
"""
import os
import pandas as pd
import wandb
from loguru import logger
from ..data_proccess import json_2_df

PROJECT_DIRS = ["model", "cache", "checkpoints"]
TASK_TYPES = ["mll_pt", "mlc_pt"]


def build_label_map(inputs: list, is_multilabel: bool = False) -> tuple:
    """Build Label Map

    :param inputs: List of Labels
    :param is_multilabel: Whether is Multi-Label
    :return: label2id, id2label
    """
    label_list = inputs
    if is_multilabel:
        label_list = [label for labels in inputs for label in labels.split(",")]
    label_list = list(set(label_list))
    label_list.sort()  # Let's sort it for determinism

    # Convert Label into ids
    label2id = dict()
    id2label = dict()
    for index, ele in enumerate(label_list):
        label2id[ele] = index
        id2label[index] = ele

    return label2id, id2label


def multilabel2list(labels: str, label2id: dict) -> list:
    """Convert Multi-Labels into Onehot Coding

    :param labels: Multi-Labels split by ","
    :param label2id: Mapping Relationship between Labels and Numbers
    :return: Onehot Coding of Multi-Labels
    """
    result = [0] * len(label2id)
    for label in labels.split(","):
        result[label2id[label]] = 1

    return result


def list2multilabel(labels: list, id2label: dict, default_label: str = "无"):
    """Convert Onehot Coding into Multi-Labels

    :param labels: Onehot Coding
    :param id2label: Mapping Relationship between Labels and Numbers
    :param default_label: Default Label
    :return: Multi-Labels split by ","
    """
    result = [id2label[index] for index, ele in enumerate(labels) if ele == 1]
    # if predict None of labels, match default label
    if len(result) < 1:
        result.append(default_label)

    return ",".join(result)


def directory_exist(directory: str) -> None:
    """Whether the directory is Exist

    :param directory: Path of Directory
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def log_init(config: dict):
    """Init Loguru

    :param config: Configuration of Loguru
    :return:
    """
    logger.add(
        config["file"],
        rotation=config["rotation"],
        compression=config["compression"]
    )


def project_init(config: dict) -> tuple:
    """Init Project : Project Directory, Log Init, Wandb Init

    :param config: Configuration of Project
    :return: Project Output Directory
    """
    # Dir Init
    output_dir = os.path.join(
        config["project_info"]["project_directory"],
        "_".join(
            [config["project_info"]["project_name"], config["project_info"]["group_name"], config["project_info"]["run_name"]]
        )
    )
    for dir_ele in PROJECT_DIRS:
        directory_exist(
            os.path.join(
                output_dir,
                dir_ele
            )
        )

    # log initial
    config["log"]["file"] = os.path.join(
        output_dir,
        "cache",
        "runtime.log"
    )
    log_init(config["log"])

    # Wandb Init
    wandb.init(
        project=config["project_info"]["project_name"],
        group=config["project_info"]["group_name"],
        name=config["project_info"]["run_name"],
        dir=output_dir,
    )

    return output_dir


def bind_data_score(inputs: list, scores: list, pred_labels: list):
    dict_score = dict()
    df_origin = json_2_df(inputs)
    df_origin["pred"] = pred_labels
    for ele in scores[0]:
        dict_score[ele["label"]] = list()
    for index, ele in enumerate(inputs):
        ele["score"] = scores[index]
        for item in scores[index]:
            dict_score[item["label"]].append(item["score"])

    df_scores = pd.DataFrame(dict_score)
    df_result = pd.concat([df_origin, df_scores], axis=1)
    return df_result


