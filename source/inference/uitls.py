# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/30 10:41
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : uitls.py
@Software   : PyCharm
@Description: 
"""
import onnxruntime
from contextlib import contextmanager
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers


def bind_score(model_result: list, id2label: dict) -> list:
    """Bind Score to Label

    :param model_result: Predict Result of Model
    :param id2label: Mapping Relationship between Labels and Numbers
    :return: List of Scores
    """
    result = list()
    for index, ele in enumerate(model_result):
        result.append(
            {
                "label": id2label[index],
                "score": float(ele),
            }
        )

    return result


def get_onnx_infer_session(model_path: str, provider: str, device_id: int = 0) -> InferenceSession:
    """Load Onnx Model From File

    :param model_path: Path of Model
    :param provider: "CUDAExecutionProvider" OR "CPUExecutionProvider"
    :param device_id: GPU ID
    :return:
    """
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    if provider == "CUDAExecutionProvider" and device_id > 0:
        session.set_providers(['CUDAExecutionProvider'], [{'device_id': device_id}])
    session.disable_fallback()
    return session