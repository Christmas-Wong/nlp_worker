# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/30 10:37
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : bert_onnx_infer.py
@Software   : PyCharm
@Description: 
"""
import torch
from tqdm import tqdm
from transformers import (
    BertTokenizer,
)
from scipy.special import expit
from ..io import json_reader
from ..core import list2multilabel, TASK_TYPES
from .uitls import bind_score, get_onnx_infer_session


class InferenceOnnxBert(object):
    def __init__(
            self,
            task: str,
            path_model: str,
            path_tokenizer: str,
            path_label2id: str,
            device: str = "cuda",
            batch_size: int = 32,
            max_seq_length: int = 512,
    ):
        super(InferenceOnnxBert, self).__init__()

        # Check Task Type
        assert task in TASK_TYPES, "task not in : " + ",".join(TASK_TYPES)
        self.task = task
        self.device = device
        if self.device == "cuda":
            self.model = get_onnx_infer_session(path_model, "CUDAExecutionProvider", 0)
        else:
            self.model = get_onnx_infer_session(path_model, "CPUExecutionProvider")
        self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        self.label2id = json_reader(path_label2id)
        self.id2label = dict()
        for key, value in self.label2id.items():
            self.id2label[value] = key
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def __bert_infer(self, inputs: list):
        # Generate Batch of Data
        data_batch = [inputs[i:i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        outputs = list()
        with torch.no_grad():
            for batch in tqdm(data_batch, desc="Bert Multi-Label Inference"):
                encoding = self.tokenizer.batch_encode_plus(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
                inputs_onnx = {k: v.cpu().detach().numpy() for k, v in encoding.items()}
                output = self.model.run(None, inputs_onnx)
                outputs += output
        return outputs

    def __score_label(self, outputs: list):
        list_scores = list()
        list_labels = list()
        if self.task == "mll_pt":
            for output in outputs:
                sigmoid = expit(output)
                pred_code = (sigmoid > 0.5).astype(int)
                list_scores += [bind_score(ele, self.id2label) for ele in sigmoid.tolist()]
                list_labels += [list2multilabel(ele, self.id2label) for ele in pred_code.tolist()]
        return list_scores, list_labels

    def infer(self, inputs: list) -> tuple:
        """Pretrain Model Infer Function

        :param inputs: List of Infer Text
        :return: Scores of Each Label & Predict Labels
        """
        # bert infer
        outputs_bert = self.__bert_infer(inputs)
        # transform outputs into scores
        scores, labels = self.__score_label(outputs_bert)
        return scores, labels