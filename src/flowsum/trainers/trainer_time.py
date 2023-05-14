# Copyright 2020 The HuggingFace Team. All rights reserved.
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

"""TimeSeq2SeqTrainer class for evaluating generation time of Seq2Seq models from the hugging face model hub."""

import time

from typing import Any, Dict, List, Optional, Tuple, Union


import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader


from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import (
    PredictionOutput,
    EvalLoopOutput,
    denumpify_detensorize,
)
from transformers.utils import logging
from transformers.trainer_pt_utils import nested_detach


logger = logging.get_logger(__name__)


class TimeSeq2SeqTrainer(Trainer):
    """Modified version of transformers.Seq2SeqTrainer.

    This trainer allows evaluation of generation time, is adapted from BOWsSeq2SeqTrainer by removing the training related functions.

    The purpose of this trainer is to evaluate the generation time of models from the hugging face model hub.
    """

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.

        Notes:
            (1) super().evaluate(), namely, Trainer.evaluate(), calls self.evaluation_loop() with
                prediction_loss_only=True if self.compute_metrics is None else None. Therefore, if we want
                to evaluate the validation set without generating tokens, but with loss/perplexity/nf_loss only,
                then we should set self.compute_metris to be None during training, and reset it to be the one
                calculating ROUGE and repetition after finishing all training.
        """

        gen_kwargs = gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().evaluate(
            eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        return super().predict(
            test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).

        Notes:
            (1) We don't need to log down gate_score values during evaluation or prediction, so we don't add
                add gate_score to the outputs as we did in self.training_step(). In this way, we don't have to
                modify the self.evaluate() function.
        """
        if not self.args.predict_with_generate:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if prediction_loss_only or self.args.prediction_loss_only:
            return (loss, None, None)

        # ================ Generate tokens ==================#
        gen_kwargs = self._gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask", None
            )

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if (
            hasattr(self.model, "encoder")
            and self.model.encoder.main_input_name != self.model.main_input_name
        ):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        t0 = time.time()
        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        gen_time = torch.tensor(
            (time.time() - t0) / generation_inputs.shape[0],
            device=generated_tokens.device,
        )  # in seconds

        # in case the batch is shorter than max length, the output should be padded
        if (
            gen_kwargs.get("max_length") is not None
            and generated_tokens.shape[-1] < gen_kwargs["max_length"]
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"]
            )
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[
            -1
        ] < (gen_kwargs["max_new_tokens"] + 1):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_new_tokens"] + 1
            )
        # ================ End of Generating tokens ==================#

        if has_labels:
            labels = inputs["labels"]
            if (
                gen_kwargs.get("max_length") is not None
                and labels.shape[-1] < gen_kwargs["max_length"]
            ):
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, (gen_kwargs["max_new_tokens"] + 1)
                )
        else:
            labels = None

        return (loss, (generated_tokens, gen_time), labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors"
                )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Notes:
            (1) There are four possible situations for outputs.predictions, depending on the arguments of
                self.prediction_step():
                1. when predict_with_generate=False or prediction_loss_only=True, outputs.predictions = None
                2. when predict_with_generate=True or prediction_loss_only=False, outputs.predictions = (generated_tokens, gen_time)

            (2) Due to the dependency on the outputs of self.prediction_step(), need to modify correspondingly
                once we change the self.prediction_step() function.

        """
        outputs = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        if isinstance(outputs.predictions, tuple):
            gen_time = outputs.predictions[1].mean()
            outputs.metrics[f"{metric_key_prefix}_gen_time"] = gen_time
            return EvalLoopOutput(
                predictions=outputs.predictions[0],
                label_ids=outputs.label_ids,
                metrics=denumpify_detensorize(
                    outputs.metrics
                ),  # should be JSON-serializable
                num_samples=outputs.num_samples,
            )
        else:
            return EvalLoopOutput(
                predictions=outputs.predictions,
                label_ids=outputs.label_ids,
                metrics=outputs.metrics,
                num_samples=outputs.num_samples,
            )
