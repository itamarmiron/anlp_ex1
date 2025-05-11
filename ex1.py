import argparse
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

import wandb
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

def data_preprocess(batch, tokenizer): #???????
    return tokenizer(batch["sentence1"], batch["sentence2"], truncation=True)


def load_data_and_split(tokenizer: BertTokenizerFast, args: argparse.Namespace):
    dataset = load_dataset("glue", "mrpc").map(lambda x: data_preprocess(x, tokenizer), batched=True) # ????
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # batch the data
    train_data, eval_data, test_data  = dataset["train"], dataset["validation"], dataset["test"]
    if args.max_train_samples != -1:
        train_data = train_data.select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        eval_data = eval_data.select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        test_data = test_data.select(range(args.max_predict_samples))
    return train_data, eval_data, test_data, data_collator

def hyperparameter_tune(parser: argparse.ArgumentParser):
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    return parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=1).numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def train_and_evaluate(args, tokenizer, train_data, eval_data, data_collator):
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        report_to="wandb",
        run_name=f"bert-mrpc-{args.lr}-{args.batch_size}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate(eval_dataset=eval_data)
    model.save_pretrained("final_model")

    # Evaluate and log to res.txt
    eval_result = trainer.evaluate(eval_dataset=eval_data)
    eval_acc = eval_result["eval_accuracy"]

    # with open("res.txt", "a") as f:
    #     f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {eval_acc:.4f}\n")


def predict(args, test_data, tokenizer):
    model = BertForSequenceClassification.from_pretrained(args.model_path)
    training_args = TrainingArguments(output_dir="./results", per_device_eval_batch_size=64)
    trainer = Trainer(model=model, args=training_args, tokenizer=tokenizer)

    predictions = trainer.predict(test_data)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()

    with open("predictions.txt", "w") as f:
        for i, p in enumerate(preds):
            s1 = test_data[i]["sentence1"]
            s2 = test_data[i]["sentence2"]
            f.write(f"{s1}###{s2}###{p}\n")


if __name__ == "__main__":
    # parse cli arguments
    args = hyperparameter_tune(argparse.ArgumentParser())

    wandb.init(
        project="ex1_anlp_final!",
        config=vars(args),
        name=f"run-lr={args.lr}-bs={args.batch_size}"
    )

    # load & split data and init tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_data, eval_data, test_data, data_collator = load_data_and_split(tokenizer, args)
    if args.do_train:
        train_and_evaluate(args, tokenizer, train_data, eval_data, data_collator)
    if args.do_predict:
        predict(args, test_data, tokenizer)