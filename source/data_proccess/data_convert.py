# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 17:35
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : data_convert.py
@Software   : PyCharm
@Description: 
"""
from pandas import DataFrame


def json_2_df(json_data: list) -> DataFrame:
    columns = list(json_data[0].keys())
    list_data = [[] for _ in columns]
    for ele in json_data:
        for key, value in ele.items():
            index = columns.index(key)
            list_data[index].append(value)

    result = DataFrame()
    for index, ele in enumerate(columns):
        result[ele] = list_data[index]
    return result


def df_2_json(df: DataFrame) -> list:
    columns = df.columns
    result = list()
    for index, row in df.iterrows():
        dict_ele = dict()
        for name in columns:
            dict_ele[name] = row[name]
        result.append(dict_ele)
    return result