# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 13:58
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : __init__.py.py
@Software   : PyCharm
@Description: 
"""
from .bert_torch_infer import InferenceBert
from .bert_onnx_infer import InferenceOnnxBert
from .uitls import bind_score