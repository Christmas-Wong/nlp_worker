# -*- coding: utf-8 -*-
"""
@Time       : 2022/5/25 14:21
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : train 
@File       : message_builder.py
@Software   : PyCharm
@Description: 
"""
from time import time, localtime, strftime


def seconds_2_hour(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def str_message(configs: dict, start_time: time(), end_time: time()) -> str:
    """Generate bot message

    :param configs: project configures
    :param start_time: start time
    :param end_time: end time
    :param metrics: model metrics
    :return: string message
    """
    outputs = [
        "懒🐶!模型训练好了，赶紧来看看，别躺着了～～～",
        "项目名称: " + '_'.join(configs['project_info'].values()),
        "开始时间: " + strftime("%Y-%m-%d %H:%M:%S", localtime(start_time)),
        "结束时间: " + strftime("%Y-%m-%d %H:%M:%S", localtime(end_time)),
        "总耗时: " + seconds_2_hour(end_time - start_time),
        "*" * 40,
    ]
    return "\n".join(outputs)
