# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/30 11:10
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : bert_onnx.py
@Software   : PyCharm
@Description: 
"""
import os
import onnx
import torch
from onnxconverter_common import float16
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification


def do_convert(
        input_huggingface_bert_model: str,
        output_onnx_model_name: str,
        num_labels: int,
        max_seq_length: int = 512,
) -> None:
    text = "test_text"
    tokenizer = BertTokenizer.from_pretrained(
        input_huggingface_bert_model,
        do_lower_case=False
    )
    config = BertConfig.from_pretrained(
        input_huggingface_bert_model,
        num_labels=num_labels,
        hidden_dropout_prob=0.3,
        problem_type="multi_label_classification"
    )
    model = BertForSequenceClassification.from_pretrained(
        input_huggingface_bert_model,
        config=config
    ).to('cpu')
    model.eval()
    encode_dict = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to("cpu")
    input_ids = encode_dict['input_ids']
    token_type_ids = encode_dict['token_type_ids']
    attention_mask = encode_dict["attention_mask"]
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask, token_type_ids),
            output_onnx_model_name,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=['output'],
            dynamic_axes={'input_ids': {0: 'batch_size'}, 'attention_mask': {0: 'batch_size'},
                          'token_type_ids': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    print("Convert into Onnx_32 Finish!")


def convert_bert_onnx(
        path_bert: str,
        path_onnx: str,
        number_label: int,
        max_seq_length: int = 512,
        fp16: bool = True
):
    """Convert BertModel into Onnx

    :param path_bert: Path of Bert
    :param path_onnx: Path of Onnx
    :param number_label: Number of Labels
    :param max_seq_length: Max Sequence length for Tokenizer
    :param fp16: Whether to get fp16 Model
    :return: None
    """
    do_convert(
        path_bert,
        os.path.join(path_onnx, "fp32", "fp32.onnx"),
        number_label,
        max_seq_length
    )
    fp32_model = onnx.load_model(os.path.join(path_onnx, "fp32", "fp32.onnx"))
    if fp16:
        fp16_model = float16.convert_float_to_float16(
            fp32_model,
            keep_io_types=True
        )
        onnx.save_model(fp16_model, os.path.join(path_onnx, "fp16", "fp16.onnx"))
        print("Convert into Onnx_16 Finish!")
