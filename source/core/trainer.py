from typing import Optional
from dataclasses import dataclass, field
from transformers import get_linear_schedule_with_warmup, AdamW

@dataclass
class TrainArguments:
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "Output Directory of this train project"}
    )
    do_train: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to Do train."}
    )
    do_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to Do Eval while Training."}
    )
    do_test: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to Do Test After Train."}
    )
    early_stop_patience: Optional[int] = field(
        default=5,
        metadata={"help": "The Patience of Early Stopping."}
    )
    do_rdrop: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to Do RDrop while Training."}
    )
    train_batch_size: Optional[bool] = field(
        default=8,
        metadata={"help": "The Batch Size of Training"}
    )
    eval_batch_size: Optional[bool] = field(
        default=8,
        metadata={"help": "The Batch Size of Eval"}
    )
    weight_decay: Optional[float] = field(
        default=0.01,
        metadata={"help": "Weight Decay"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Gradient Accumulation Steps"}
    )


class TrainerBert(object):
    def __init__(self, model, tokenizer, optimizer, scheduler, train_args):
        super(TrainerBert, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = train_args
        self.train_loss = 0.0
        self.train_step = 0

    def __train_epoch() -> None:
        for step, (inputs_ids, token_type_ids, attention_mask, labels) in tqdm(epoch_iterator):
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            model.train()
            outputs = model(
                    inputs_ids,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_mask,
            )
            loss = outputs[0].mean()
            self.train_loss += loss.item()


