#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

#################
# Added by Aslihan
import json
import torch
import pickle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,classification_report,confusion_matrix

# from transformers import XLMTokenizer, XLMWithLMHeadModel
# xml_tokenizer = XLMTokenizer.from_pretrained("xlm-clm-enfr-1024")
# xml_model = XLMWithLMHeadModel.from_pretrained("xlm-clm-enfr-1024")
# from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertModel.from_pretrained("bert-base-multilingual-cased")

# BERT_MODEL_NAME = 'bert-base-cased'
# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

################

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    n_classes: int = field(
                default=2
    )

    train_file: Optional[str] = field(
        default="./emoji_sentiment_data/splitted_processed_data/2_classes/train.json",
        metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default="./emoji_sentiment_data/splitted_processed_data/2_classes/validation.json",
        metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default="./emoji_sentiment_data/splitted_processed_data/2_classes/test.json",
                                     metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            if not self.dataset_name == "bsc/tecla":
                raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        #print(model_args)
        #print(data_args)
        #print(training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name == "BSC-TeMU/mesinesp":
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, cache_dir=model_args.cache_dir)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        print(data_args.train_file)

        data_args.train_file = f"./emoji_sentiment_data/splitted_processed_data/{data_args.n_classes}_classes/train.json"
        data_args.test_file = f"./emoji_sentiment_data/splitted_processed_data/{data_args.n_classes}_classes/test.json"
        data_args.val_file =  f"./emoji_sentiment_data/splitted_processed_data/{data_args.n_classes}_classes/validation.json"
        
        data_files = {"train": data_args.train_file, "val": data_args.validation_file, "test": data_args.test_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            train = json.load(open(data_files["train"]))
            val = json.load(open(data_files["val"]))
            test = json.load(open(data_files["test"]))
            unique_label_list = ["positive","negative"]

            #datasets = load_dataset("mesinesp.py", data_files=data_files, cache_dir=model_args.cache_dir)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Label
    is_regression = False
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"  # semantic texture similarity
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = datasets['train'].features['label'].feature.names
        label_list.sort()
        num_labels = datasets['train'].features['label'].feature.num_classes
        datasets = datasets.map(
            lambda x: {"label": [1 if i in x["label"] else 0 for i in range(num_labels)]})
        true_labels = datasets["test"]["label"]

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    id_to_label=None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        id_to_label = {i: v for i, v in enumerate(label_list)}
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True,
                           return_token_type_ids=False)
        return result

        
        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #    result["label"] = [(label_to_id[l] if l != -1 else -1) for sublist in examples["label"] for l in sublist]


    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)

    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
        y_pred_tensor = torch.from_numpy(y_pred)
        y_true_tensor = torch.from_numpy(y_true)
        if sigmoid:
            y_pred_tensor = y_pred_tensor.sigmoid()
        y_pred = (y_pred_tensor > thresh).float().numpy()
#   [0]
        if not os.path.exists(f"{training_args.output_dir}/results"):
            os.makedirs(f"{training_args.output_dir}/results")

        classification_report_ = classification_report(y_true=y_true, y_pred=y_pred,zero_division=0,output_dict=True )

        with open(f"{training_args.output_dir}/results/eval_classification_report.pickle", "wb+") as file:
            pickle.dump(classification_report_,file,protocol=pickle.HIGHEST_PROTOCOL)

        return {'accuracy_thresh': ((y_pred_tensor > thresh) == y_true_tensor.bool()).float().mean().item(),\
                'accuracy_score':accuracy_score(y_true=y_true, y_pred=y_pred),\
                'recall_score':recall_score(y_true=y_true, y_pred=y_pred, average='samples',zero_division=0),\
                'f1_score':f1_score(y_true=y_true, y_pred=y_pred, average='samples',zero_division=0),\
                'precision_score': precision_score(y_true=y_true, y_pred=y_pred, average='samples',zero_division=0)
                }

        #return ((y_pred > thresh) == y_true.bool()).float().mean().item()

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        #accuracy_thresh
        accuracy_dict = accuracy_thresh(predictions, labels)
        return {'accuracy_thresh': accuracy_dict['accuracy_thresh'],
                'accuracy_score': accuracy_dict['accuracy_score'],
                'recall_score': accuracy_dict['recall_score'],
                'f1_score': accuracy_dict['f1_score'],
                'precision_score': accuracy_dict['precision_score']
                }

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    class MultilabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.float().view(-1, self.model.config.num_labels))
            return (loss, outputs) if return_outputs else loss

    # Initialize our Trainer
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:

        logger.info("*** Evaluate ***")
        # force evaluation on test set
        eval_datasets = [eval_dataset, predict_dataset]
        types = ['eval', 'predict']

        for eval_dataset, type in zip(eval_datasets, types):
            metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=type)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics(type, metrics)
            trainer.save_metrics(type, metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = torch.from_numpy(predictions)
            predictions = predictions.sigmoid()
            predictions = predictions.numpy()
            #predictions = (predictions > 0.5).float().numpy()

            #predictions_argsort = np.argsort(predictions, axis=1)[::-1]
            #output_predict_file = os.path.join(training_args.output_dir, f"predictions.txt")

            if trainer.is_world_process_zero():
                # to_pickle
                #save sorted predictions
                if not os.path.exists(f"{training_args.output_dir}/results"):
                    os.makedirs(f"{training_args.output_dir}/results")

                with open(f"{training_args.output_dir}/results/predictions.pickle", "wb+") as file:
                    pickle.dump(predictions,file,protocol=pickle.HIGHEST_PROTOCOL)
                #save decs_codes
                with open(f"{training_args.output_dir}/results/prediction_decs_codes.pickle", "wb+") as file:
                    pickle.dump(label_list,file,protocol=pickle.HIGHEST_PROTOCOL)
                # real
                with open(f"{training_args.output_dir}/results/real_labels.pickle", "wb+") as file:
                    pickle.dump(np.array(true_labels),file,protocol=pickle.HIGHEST_PROTOCOL)

                #with open(output_predict_file, "w") as writer:
                #    logger.info(f"***** Predict results {task} *****")
                #    writer.write("index\tprediction\n")
                #    for index, item_list in predictions_top_10:
                #        writer.write(f"{index}\t{item_list}\n")

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tags": "text-classification"}
        if data_args.task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = data_args.task_name
            kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()