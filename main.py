# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 14:03
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : main.py
@Software   : PyCharm
@Description: 
"""
import argparse
from pprint import pprint
from source.io.reader import yaml_reader
from source.train_pt.multi_label import run as mll_pt_train

DICT_TASK = {
    "mll_pt": mll_pt_train
}


def main():
    # Get Param From Shell
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    # Read Train Config
    config = yaml_reader(args.config)
    # Print Config
    pprint(config)

    # Run Task
    DICT_TASK[config["task"]](config)


if __name__ == "__main__":
    main()