# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 15:28
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : bert.py
@Software   : PyCharm
@Description: 
"""
import torch
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)
from ..io import json_reader
from ..core import list2multilabel


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


class InferenceBert(object):
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
        super(InferenceBert, self).__init__()

        # Make sure task_type is in within range
        list_task = ["mll_pt", "mlc_pt"]
        assert task in list_task, "task not in : " + ",".join(list_task)
        self.task = task

        self.model = BertForSequenceClassification.from_pretrained(path_model)
        self.tokenizer = BertTokenizer.from_pretrained(path_tokenizer)
        self.label2id = json_reader(path_label2id)
        self.id2label = dict()
        for key, value in self.label2id.items():
            self.id2label[value] = key
        self.batch_size = batch_size
        self.device = device
        self.max_seq_length = max_seq_length

    def __bert_infer(self, inputs: list):
        # Generate Batch of Data
        data_batch = [inputs[i:i + self.batch_size] for i in range(0, len(inputs), self.batch_size)]
        self.model.to(self.device)
        self.model.eval()
        outputs = list()
        with torch.no_grad():
            for batch in tqdm(data_batch, desc="Bert Multi-Label Inference"):
                text_batch = [ele["text"] for ele in batch]
                encoding = self.tokenizer.batch_encode_plus(
                    text_batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                token_type_ids = encoding["token_type_ids"].to(self.device)
                output = self.model(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                outputs.append(output)
        return outputs

    def __score_label(self, outputs: list):
        list_scores = list()
        list_labels = list()
        if self.task == "mll_pt":
            for output in outputs:
                sigmoid = output.logits.sigmoid().detach().cpu().numpy()
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
