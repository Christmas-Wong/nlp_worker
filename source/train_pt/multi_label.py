# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 14:04
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : nlp_worker 
@File       : multi_label.py
@Software   : PyCharm
@Description: 
"""
import os
import time
import torch
import numpy as np
from loguru import logger
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    Trainer,
    BertConfig,
    BertTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    default_data_collator,
    DataCollatorWithPadding,
    BertForSequenceClassification,
)
from ..inference import InferenceBert
from ..message import FeiShuApp, str_message
from ..evaluate.classification_evaluate import report_confusion
from ..io import (
    json_writer,
    yaml_writer,
    jsonl_reader,
)
from ..core import (
    project_init,
    multilabel2list,
    build_label_map,
    bind_data_score,
    DataTrainingArguments,
)


os.environ["WANDB_START_METHOD"] = "thread"


def run(config: dict):
    start_time = time.time()

    # Project Init
    output_dir = project_init(config)

    training_args = TrainingArguments(**config["train_arguments"])
    training_args.output_dir = os.path.join(
        output_dir,
        "checkpoints",
    )
    data_args = DataTrainingArguments(**config["data_training_arguments"])

    # Load Dataset
    data_files = {
        "train": data_args.train_file,
        "valid": data_args.validation_file,
    }
    file_extension = data_args.train_file.split(".")[-1]
    raw_datasets = load_dataset(file_extension, data_files=data_files)

    # Convert Label into ids
    label2id, id2label = build_label_map(
        raw_datasets["train"].unique("label"),
        is_multilabel=True
    )
    logger.info(label2id)

    # Load Model and Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["pre_train_model"]["tokenizer"])
    bert_config = BertConfig.from_pretrained(
        config["pre_train_model"]["model"],
        num_labels=len(label2id),
        problem_type="multi_label_classification"
    )
    bert_config.hidden_dropout_prob = config["train"]["hidden_dropout"]
    bert_config.attention_probs_dropout_prob = config["train"]["attention_dropout"]
    model = BertForSequenceClassification.from_pretrained(
        config["pre_train_model"]["model"],
        config=bert_config
    )
    model.config.label2id = label2id
    model.config.id2label = id2label

    # Data Process
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples["text"],
            padding=data_args.pad_to_max_length,
            max_length=data_args.max_seq_length,
            truncation=True
        )
        # Map labels to IDs
        result["label"] = [multilabel2list(item, label2id) for item in examples["label"]]
        return result
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    # Define Metric
    def compute_metrics(p):
        pred, labels = p
        pred = torch.sigmoid(torch.from_numpy(pred).to("cuda")).detach().cpu().numpy()
        prediction = list()
        for item in pred:
            prediction.append((item > 0.5).astype(float))
        pred = np.array(prediction)

        precision, recall, fscore, _ = precision_recall_fscore_support(labels, pred, average="macro")
        accuracy = accuracy_score(labels, pred)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": fscore,
        }

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["valid"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["train"]["early_stop"])],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model(
            os.path.join(
                output_dir,
                "model"
            )
        )
        json_writer(
            label2id,
            os.path.join(output_dir, "cache", "label2id.json")
        )
        json_writer(
            id2label,
            os.path.join(output_dir, "cache", "id2label.json")
        )

    if training_args.do_eval:
        list_test_data = jsonl_reader(data_args.test_file)
        inference = InferenceBert(
            task=config["task"],
            path_model=os.path.join(output_dir, "model"),
            path_tokenizer=os.path.join(output_dir, "model"),
            path_label2id=os.path.join(output_dir, "cache", "label2id.json"),
            device=config["device"],
            max_seq_length=data_args.max_seq_length,
        )
        scores, pred_labels = inference.infer(list_test_data)
        true = [multilabel2list(ele["label"], label2id) for ele in list_test_data]
        pred = [multilabel2list(ele, label2id) for ele in pred_labels]
        report = report_confusion(
            true,
            pred,
            label2id.keys(),
            os.path.join(output_dir, "cache"),
            config["task"]
        )
        df_scores = bind_data_score(list_test_data, scores, pred_labels)
        df_scores.to_csv(
            os.path.join(
                output_dir,
                "cache",
                "predict_test.csv"
            ),
            index=False
        )
        yaml_writer(
            config,
            os.path.join(
                output_dir,
                "cache",
                "config.yml"
            )
        )
        end_time = time.time()
        feishu_app = FeiShuApp(
            app_id=config["message"]["feishu_bot"]["app_id"],
            app_secret=config["message"]["feishu_bot"]["app_secret"],
        )
        feishu_app.send_message(
            receiver_phone=config["message"]["feishu_bot"]["to_user_phone"],
            message=str_message(config, start_time, end_time) + "\n" + report
        )

    if training_args.do_predict:
        train_test = jsonl_reader(data_args.train_file) + jsonl_reader(data_args.test_file)
        inference = InferenceBert(
            task=config["task"],
            path_model=os.path.join(output_dir, "model"),
            path_tokenizer=os.path.join(output_dir, "model"),
            path_label2id=os.path.join(output_dir, "cache", "label2id.json"),
            device=config["device"],
            max_seq_length=data_args.max_seq_length,
        )
        scores, pred_labels = inference.infer(train_test)
        df_scores = bind_data_score(train_test, scores, pred_labels)
        df_scores.to_csv(
            os.path.join(
                output_dir,
                "cache",
                "predict_train_test.csv"
            ),
            index=False
        )

