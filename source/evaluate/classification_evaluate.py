# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 15:56
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : report_confusion_prediction.py
@Software   : PyCharm
@Description: 
"""
import os
from loguru import logger
from prettytable import PrettyTable
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix
)
from ..io import report_saver, json_writer, confusion_2_csv


def pretty_report(report: dict):
    result = PrettyTable()
    result.field_names = ["Class", "precision", "recall", "f1-score", "support"]
    for key, value in report.items():
        if key == "accuracy":
            result.add_row([key, 0, 0, value, report["weighted avg"]["support"]])
            continue
        result.add_row([key, str(value["precision"]), str(value["recall"]), str(value["f1-score"]), str(value["support"])])
    return result


def report_confusion(
        true: list,
        pred: list,
        labels_names: list,
        output_dir: str,
        task: str = "mlc"
) -> str:
    """ Build Report & Confusion Matrix

    :param true: True Data
    :param pred: Prediction Data
    :param labels_names: All Labels
    :param output_dir: Output Directory
    :param task: Task Type
    :return: Report txt
    """
    report_dict = classification_report(
        true,
        pred,
        target_names=labels_names,
        output_dict=True,
        digits=4
    )
    report_txt = classification_report(
        true,
        pred,
        target_names=labels_names,
        output_dict=False,
        digits=4
    )
    report_table = pretty_report(report_dict)
    logger.info("\n" + str(report_table))

    if task == "mlc":
        confusion = confusion_matrix(true, pred)
    elif task == "mll_pt":
        confusion = multilabel_confusion_matrix(true, pred)
        counter = 0
        for key in labels_names:
            confusion_2_csv(
                confusion[counter],
                ["否", key],
                os.path.join(
                    output_dir,
                    key + "_confusion_matrix.csv"
                )
            )

    logger.info("\n" + str(confusion))

    report_saver(
        report_table,
        report_txt,
        output_dir
    )

    return report_txt
