#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.

Reference: https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization
"""


import os
import logging
import sys
import yaml
import json
import argparse
from functools import partial

import nltk
import datasets
import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_offline_mode
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback

from flowsum.utils.train_utils import (
    load_custom_model,
    get_trainer_cls,
    ModelArguments,
    DataTrainingArguments,
    MiscArguments,
    CustomSeq2SeqTrainingArguments,
    CUSTOM_MODELS,
)
from flowsum.utils.eval_utils import (
    compute_metrics,
    evaluate_rouge_on_file,
    evaluate_repetition_on_file,
    evaluate_bertscore_on_file,
)
from flowsum.utils.data_utils import (
    preprocess_data_function,
    get_text_summary_column,
    get_raw_datasets,
)
from flowsum.utils import log_utils
from flowsum.trainers.callbacks import VisualTensorBoardCallback

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main(args):
    """
    Args:
        args: should be a dictionary with model_args, data_args, and training_args, which are all dictionaries themselves.
            Can be obtained by reading the .yml configuration file.
    """
    model_args = ModelArguments(**args["model_args"])
    data_args = DataTrainingArguments(**args["data_args"])
    misc_args = MiscArguments(**args["miscellaneous_args"])

    if misc_args.trainer_name.startswith("seq2seq"):
        training_args = Seq2SeqTrainingArguments(**args["training_args"])
    else:
        training_args = CustomSeq2SeqTrainingArguments(**args["training_args"])
    if misc_args.generate_pseudo_labels:
        training_args.do_predict = True

    if not (args["training_args"]["do_eval"] or args["training_args"]["do_train"]):
        training_args.do_eval = False

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # Setup logging
    log_path = os.path.join(training_args.output_dir, "log.txt")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )  # the config will be added to the root logger automatically
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # set up loggers for scripts in the flowsum module
    log_utils.set_verbosity(log_level)
    log_utils.enable_default_handler()
    log_utils.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Args: {json.dumps(args, indent=4)}")
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Model arguments: {model_args}")

    if model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        assert (
            data_args.source_prefix == "summarize: "
        ), "You're running a t5 model but didn't provide a source prefix 'summarize: '"

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = get_raw_datasets(data_args=data_args, model_args=model_args)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # config_name here can also be config_path, such as config.json. The from_pretrained() method also takes .json as input.
    if model_args.model_name_or_path in CUSTOM_MODELS:
        assert (
            data_args.add_bows
        ), "When running custom models, `data_args.add_bows` should be True."
        assert (
            model_args.config_name is not None
        ), "When running custom models without pretrained or finetuned weights, `model_args.config_name` cannot be None."
        assert (
            model_args.tokenizer_name is not None
        ), "When running custom models, `model_args.tokenizer_name` cannot be None."

    model_ckpt_path = None
    if (
        (not training_args.do_train)
        and training_args.do_predict
        and training_args.resume_from_checkpoint is not None
    ) or (
        data_args.distilled_only and training_args.resume_from_checkpoint is not None
    ):
        assert os.path.isdir(
            training_args.resume_from_checkpoint
        ), "`training_args.resume_from_checkpoint` should be a directory."
        config_path = os.path.join(training_args.resume_from_checkpoint, "config.json")
        model_ckpt_path = os.path.join(
            training_args.resume_from_checkpoint, "pytorch_model.bin"
        )
        assert os.path.exists(config_path) and os.path.exists(
            model_ckpt_path
        ), "`training_args.resume_from_checkpoint` should contain 'config.json' and 'pytorch_model.bin'."

        model_args.config_name = config_path

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )  # when running custom models with pretrained or finetuned weights, config will not be used.

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.model_name_or_path.startswith("nf-bert2bert"):
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
    if model_args.model_name_or_path.startswith("nf-gpt2"):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # ref: https://github.com/huggingface/transformers/issues/12594#issuecomment-877358955

    if model_args.model_name_or_path in CUSTOM_MODELS:
        model = load_custom_model(
            model_name=model_args.model_name_or_path,
            config=config,
            tokenizer=tokenizer,
            model_args=model_args,
            model_ckpt_path=model_ckpt_path,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        data_args.add_bows = False

    logger.info(f"Total number of model parameters: {model.num_parameters()}")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if (
        model.config.decoder_start_token_id is None
        and "gpt" not in model_args.model_name_or_path
    ):
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    text_column, summary_column = get_text_summary_column(
        data_args=data_args, column_names=column_names
    )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    preprocess_function = partial(
        preprocess_data_function,
        text_column=text_column,
        summary_column=summary_column,
        prefix=prefix,
        tokenizer=tokenizer,
        max_source_length=data_args.max_source_length,
        max_target_length=max_target_length,
        padding=padding,
        ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
        add_bows=data_args.add_bows,
    )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if misc_args.generate_pseudo_labels:
            predict_dataset = raw_datasets["train"]
        else:
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Callbacks
    # the callbacks used by the trainer will be the ones defined here + the default callbacks
    # the default callbacks are: [DefaultFlowCallback, TensorBoardCallback, ProgressCallback]
    # EarlyStoppingCallback reference: https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/trainer_callback.py#L524
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=misc_args.early_stopping_patience,
            early_stopping_threshold=misc_args.early_stopping_threshold,
        )
    ]
    if (
        hasattr(training_args, "visualize_latent_dist")
        and training_args.visualize_latent_dist
    ):
        callbacks.append(VisualTensorBoardCallback)

    # Initialize our Trainer
    if model_args.model_name_or_path in CUSTOM_MODELS:
        assert (
            misc_args.trainer_name != "seq2seq"
        ), "Custom models should use custom trainers, such as 'bows_seq2seq' and 'lag_bows_seq2seq'."
    trainer_cls = get_trainer_cls(trainer_name=misc_args.trainer_name)
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,  # during training, we don't want to generate tokens
        callbacks=callbacks,
    )

    # More manipulations on callbacks
    # if we add VisualTensorBoardCallback, then we need to remove the default TensorBoardCallback.
    if (
        hasattr(training_args, "visualize_latent_dist")
        and training_args.visualize_latent_dist
    ):
        trainer.remove_callback(TensorBoardCallback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None and (
            not data_args.distilled_only
        ):
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Reset trainer's compute_metrics for evaluation and prediction with generate
    if training_args.predict_with_generate:
        trainer.compute_metrics = partial(
            compute_metrics,
            tokenizer=tokenizer,
            metric=evaluate.load("rouge"),
            ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
        )

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else (
            training_args.generation_num_beams
            if training_args.generation_num_beams is not None
            else model.config.num_beams
        )
    )
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=max_length,
            num_beams=num_beams,
        )
        metrics = (
            predict_results.metrics
        )  # this metric is given by compute_metrics() based on truncated version
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions = [
                    pred.strip().replace("\n", " ") for pred in predictions
                ]  # the generated text of NFBart contains "\n" while Bart doesn't.
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.txt"
                )
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

                # evaluate ROUGE on the full generated text
                if data_args.max_predict_samples is None:
                    rouge_scores = evaluate_rouge_on_file(
                        prediction_path=output_prediction_file,
                        dataset=raw_datasets["test"],
                        text_column=text_column,
                        summary_column=summary_column,
                    )
                    with open(
                        os.path.join(training_args.output_dir, "prediction_rouge.json"),
                        "w",
                    ) as fp:
                        json.dump(rouge_scores, fp)

                # evaluate repetition on the generated text (can be selected ones)
                rep_scores = evaluate_repetition_on_file(
                    prediction_path=output_prediction_file, bandwidth=16, n=[1, 2, 3, 4]
                )
                with open(
                    os.path.join(
                        training_args.output_dir, "prediction_repetition.json"
                    ),
                    "w",
                ) as fp:
                    json.dump(rep_scores, fp)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train an nf-enhanced summarization model.")
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=[5, 6],
        help="GPU ids. (ex. 3 4 5 6)",
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        default="./experiments/train/example.yml",
        help="The configuration file, including model_args, data_args, and training_args.",
    )
    cmd_args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(i) for i in cmd_args.gpu_ids])

    with open(cmd_args.config_file_path, "r") as fp:
        args = yaml.safe_load(fp)

    main(args)
