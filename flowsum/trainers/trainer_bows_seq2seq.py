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

import math
import sys
import os
import time
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
from matplotlib.figure import Figure

from tqdm.auto import tqdm

# Integrations must be imported before ML frameworks:
from transformers.integrations import hp_params

import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer import Trainer
from transformers.trainer_utils import (
    PredictionOutput,
    EvalLoopOutput,
    ShardedDDPOption,
    TrainOutput,
    HPSearchBackend,
    IntervalStrategy,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    logging,
    is_sagemaker_mp_enabled,
    is_apex_available,
    is_torch_tpu_available,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    nested_detach,
)
from transformers.modeling_utils import unwrap_model
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer_callback import TrainerState
from transformers.pytorch_utils import is_torch_less_than_1_11

from flowsum.nf.model import DiagGaussian
from flowsum.utils.visual_utils import visualize_latent_and_prior

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)


TRAINER_STATE_NAME = "trainer_state.json"


class BOWsSeq2SeqTrainer(Trainer):
    """Modified version of transformers.Seq2SeqTrainer.

    This trainer takes 'decoder_bows' in the inputs, and allows visualization of the latent distribution
        in Tensorboard if misc_args.visualize_latent_dist = True.
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
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
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
                if (
                    isinstance(outputs, dict)
                    and "perplexity" in outputs.keys()
                    and "nf_loss" in outputs.keys()
                ):
                    perplexity = outputs["perplexity"].mean()
                    nf_loss = outputs["nf_loss"].mean()
                else:
                    perplexity = None
                    nf_loss = None
            else:
                loss = None

        if not self.args.predict_with_generate or prediction_loss_only:
            if perplexity is not None and nf_loss is not None:
                logits = (perplexity, nf_loss)
                logits = nested_detach(logits)
                return (loss, logits, None)
            else:
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
        # ========= Added by Yu ===========
        if "decoder_bows" in inputs:
            gen_kwargs["decoder_bows"] = inputs.get("decoder_bows", None)
        # =================================

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

        if perplexity is not None and nf_loss is not None:
            logits = (
                generated_tokens,  # be of batch_size
                gen_time,  # one single value
                perplexity,  # one single value
                nf_loss,  # one single value
            )  # follow Trainer.prediction_step(), make it into a tuple
            logits = nested_detach(logits)
            return (loss, logits, labels)
        else:
            return (loss, generated_tokens, labels)

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
                1. when predict_with_generate=False or prediction_loss_only=True, and no perplexity or nf_loss
                    in the model output, outputs.predictions = None
                2. when predict_with_generate=False or prediction_loss_only=True, and there are perplexity
                    and nf_loss in the model output, outputs.predictions = (perplexity, nf_loss)
                3. when predict_with_generate=True or prediction_loss_only=False, and no perplexity or nf_loss
                    in the model output, outputs.predictions = generated_tokens
                4. when predict_with_generate=True or prediction_loss_only=False, and there are perplexity
                    and nf_loss in the model output, outputs.predictions = (generated_tokens, gen_time, perplexity, nf_loss)

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
        if isinstance(
            outputs.predictions, tuple
        ):  # when it contains perplexity and nf_loss
            perplexity = outputs.predictions[-2].mean()
            nf_loss = outputs.predictions[-1].mean()
            outputs.metrics[f"{metric_key_prefix}_perplexity"] = perplexity
            outputs.metrics[f"{metric_key_prefix}_nf_loss"] = nf_loss
            if len(outputs.predictions) == 4:
                gen_time = outputs.predictions[-3].mean()
                outputs.metrics[f"{metric_key_prefix}_gen_time"] = gen_time
            return EvalLoopOutput(
                predictions=outputs.predictions[0]
                if len(outputs.predictions) == 4
                else None,  # the generated tokens if outputs.predictions = (generated_tokens, gen_time, perplexity, nf_loss)
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

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        """Modified version of Trainer._inner_training_loop().

        The main idea is to add perplexity and nf_loss to the metrics and to Tensorboard.

        To fully understand the coding logic, it's helpful to read the trainer.py, the TensorBoardCallback
            class, and the CallbackHandler class. The following is a brief illustration.
            1. In Trainer.__init__(), the TensorBoardCallback will be added by the
                get_reporting_integration_callbacks() function. We can use
                `trainer.callback_handler.callback_list()` in the debug mode to check the available
                callbacks.
            2. There are two places where TensorBoardCallback will be called for logging.
                1. Inside `self._inner_training_loop()`, `self.log(metrics)` calls `on_log()`. This one
                    is performed after training, only slightly before `on_train_end()`. It will not log
                    down the progress of the metrics, but only the final output.
                2. Inside `_inner_training_loop()`, `_maybe_log_save_evaluate()` calls `on_log()`. This
                    one is performed after training each step, right after `on_step_end()`. It will log
                    down the progress. Therefore, we need to customize `_maybe_log_save_evaluate()`.

        Since this function calls self.training_step() to get the loss, we also modify the self.training_step()
            so that it outputs perplexity and nf_loss as well.

        References:
            (1) trainer.py: https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/trainer.py
            (2) TensorBoardCallback: https://github.com/huggingface/transformers/blob/820c46a707ddd033975bc3b0549eea200e64c7da/src/transformers/integrations.py#L556
            (3) CallbackHandler: https://github.com/huggingface/transformers/blob/v4.26.0/src/transformers/trainer_callback.py#L290
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Set up the example used for visualization
        #   in order to track the changes, we need to fix the examples.
        if (
            self.args.visualize_latent_dist
            and "decoder_bows" in train_dataloader.dataset.features
        ):
            self._visual_example_decoder_bows = iter(train_dataloader).next()[
                "decoder_bows"
            ][
                : args.num_visualize_examples
            ]  # (args.num_visualize_examples, vocab_size) if args.num_visualize_examples <= _train_batch_size, o.w., (batch_size, vocab_size)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = (
                len_dataloader // args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = (
                    self.num_examples(train_dataloader) * args.num_train_epochs
                )
        elif (
            args.max_steps > 0
        ):  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self,
                num_training_steps=max_steps,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch
                    )
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches"
                    )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if (
            args.visualize_latent_dist
        ):  # when the callback_handler calls on_visualize, it requires every callback inside it to have the attribute.
            for callback in self.callback_handler.callbacks:
                if not hasattr(callback, "on_visualize"):
                    callback.on_visualize = lambda *args, **kwargs: None

        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        tr_ppl = torch.tensor(0.0).to(args.device)
        tr_nf_loss = torch.tensor(0.0).to(args.device)
        tr_gate_score = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._total_ppl_scalar = 0.0
        self._total_nf_loss_scalar = 0.0
        self._total_gate_score_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(
                train_dataloader.dataset, IterableDatasetShard
            ):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(
                    train_dataloader, [args.device]
                ).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        (
                            tr_loss_step,
                            tr_ppl_step,
                            tr_nf_loss_step,
                            tr_gate_score_step,
                        ) = self.training_step(model, inputs)
                else:
                    (
                        tr_loss_step,
                        tr_ppl_step,
                        tr_nf_loss_step,
                        tr_gate_score_step,
                    ) = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                    tr_ppl += tr_ppl / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                    tr_nf_loss += tr_nf_loss / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                    tr_gate_score += tr_gate_score / (
                        1 + self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    tr_loss += tr_loss_step
                    tr_ppl += tr_ppl_step
                    tr_nf_loss += tr_nf_loss_step
                    tr_gate_score += tr_gate_score_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if (
                        args.max_grad_norm is not None
                        and args.max_grad_norm > 0
                        and not self.deepspeed
                    ):
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce(
                                    "sum", gradients, scale=1.0 / xm.xrt_world_size()
                                )
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer)
                                if self.use_apex
                                else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(
                        tr_loss,
                        tr_ppl,
                        tr_nf_loss,
                        tr_gate_score,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                    )
                    self._maybe_visualize()
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss,
                tr_ppl,
                tr_nf_loss,
                tr_gate_score,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
            )
            self._maybe_visualize()

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        self._total_ppl_scalar += tr_ppl.item()
        train_ppl = self._total_ppl_scalar / self.state.global_step
        self._total_nf_loss_scalar += tr_nf_loss.item()
        train_nf_loss = self._total_nf_loss_scalar / self.state.global_step
        self._total_gate_score_scalar += tr_gate_score.item()
        train_gate_score = self._total_gate_score_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        metrics["train_ppl"] = train_ppl
        metrics["train_nf_loss"] = train_nf_loss
        metrics["train_gate_score"] = train_gate_score

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """Modified version of Trainer.training_step().

        The main modification is to add perplexity and nf_loss to the output as needed
            self.in _inner_training_loop().

        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps
            )
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            # a modification of super().compute_loss(), the reason for not modifying compute_loss() is
            # that prediction_step() also uses compute_loss().
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs)

            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                if (
                    unwrap_model(model)._get_name()
                    in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
                ):
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # ============== add perplexity and nf_loss ==============#
            if (
                isinstance(outputs, dict)
                and "perplexity" in outputs.keys()
                and "nf_loss" in outputs.keys()
            ):
                perplexity = outputs["perplexity"]
                nf_loss = outputs["nf_loss"]
            else:
                perplexity = None
                nf_loss = None

            # ============== add gate_score ==============#
            if isinstance(outputs, dict) and "gate_score" in outputs.keys():
                gate_score = outputs["gate_score"]
            else:
                gate_score = None

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            perplexity = perplexity.mean()
            nf_loss = nf_loss.mean()
            gate_score = gate_score.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach(), perplexity.detach(), nf_loss.detach(), gate_score.detach()

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        tr_ppl,
        tr_nf_loss,
        tr_gate_score,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
    ):
        """Modified version of Trainer._maybe_log_save_evaluate().

        The main modification is to add perplexity and nf_loss to the logs, so that we can track the
            changes in Tensorboard.

        This function is called inside self._inner_training_loop().

        Notes:
            (1) In Trainer.evaluate(), after calculating all the metrics,
                - it calls self.log(output.metrics), which logs down the metrics to Tensorboard, so we don't
                    have to explicitly call self.callback_handler.on_log().
                - it calls self.control = self.callback_handler.on_evaluate(), so we don't have to call
                    self.callback_handler.on_evaluate() again.
        """
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_ppl_scalar = self._nested_gather(tr_ppl).mean().item()
            tr_nf_loss_scalar = self._nested_gather(tr_nf_loss).mean().item()
            tr_gate_score_scalar = self._nested_gather(tr_gate_score).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            tr_ppl -= tr_ppl
            tr_nf_loss -= tr_nf_loss
            tr_gate_score -= tr_gate_score

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["ppl"] = round(
                tr_ppl_scalar / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["nf_loss"] = round(
                tr_nf_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["gate_score"] = round(
                tr_gate_score_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._total_ppl_scalar += tr_ppl_scalar
            self._total_nf_loss_scalar += tr_nf_loss_scalar
            self._total_gate_score_scalar += tr_gate_score_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def _maybe_visualize(self):
        """Modified version of BOWsSeq2SeqTrainer._maybe_log_save_evaluate().

        The main modification is to add visualization

        This function is called inside self._inner_training_loop().
        """
        if (
            self.args.visualize_latent_dist
            and self.args.visualization_strategy == IntervalStrategy.STEPS
            and self.args.visual_steps > 0
            and self.state.global_step % self.args.visual_steps == 0
        ):
            figures = self.visualize()
            self.callback_handler.call_event(
                "on_visualize", self.args, self.state, self.control, figures=figures
            )

    def visualize(self) -> List[Figure]:
        """"""
        assert (
            self._visual_example_decoder_bows.dim() == 2
        ), "self._visual_example_decoder_bows should have 2 dimensions with shape (batch_size, vocab_size)."
        visual_example_decoder_bows = self._visual_example_decoder_bows.to(
            device=self.model.device
        )  # (args.num_visualize_examples, vocab_size)
        args = self.args
        nf_latent_model = self.model.nf_latent_model

        nf_input = (
            torch.mm(
                visual_example_decoder_bows,
                self.model.embeddings
                if hasattr(self.model, "embeddings")  # NFBert2bert, NFPegasus, NFT5
                else self.model.model.shared.weight,  # NFBart
            )  # (args.num_visualize_examples, embed_size)
            if self.model.q_input_type == "avg_embed"
            else visual_example_decoder_bows  # (args.num_visualize_examples, vocab_size)
        )
        mu, log_sigma = torch.split(
            nf_latent_model.q_net(nf_input),
            split_size_or_sections=nf_latent_model.nf_latent_size,
            dim=-1,
        )  # both are (args.num_visualize_examples, latent_size)

        z, _ = DiagGaussian(
            mu=mu.repeat_interleave(
                repeats=args.num_visualize_samples_per_example, dim=0
            ),  # (args.num_visualize_samples_per_example * args.num_visualize_examples, latent_size)
            log_sigma=log_sigma.repeat_interleave(
                repeats=args.num_visualize_samples_per_example, dim=0
            ),  # (args.num_visualize_samples_per_example * args.num_visualize_examples, latent_size)
        )  # (args.num_visualize_samples_per_example * args.num_visualize_examples, latent_size)
        z0 = z[:, :2]  # only keep the first two dimensions for plotting

        # note: we need all latent_size elements for transformation
        for transform in nf_latent_model.nf_model.transforms:
            z = transform(
                z
            )  # (args.num_visualize_samples_per_example * args.num_visualize_examples, latent_size)

        z_prior = torch.randn_like(
            z0
        )  # (args.num_visualize_samples_per_example * args.num_visualize_examples, 2), we only need the first two dimensions for plotting, so only generate 2

        # Plot the first two dimensions
        figures = []
        for i, (_z0, _z, _z_prior) in enumerate(
            zip(
                torch.split(
                    z0,
                    split_size_or_sections=args.num_visualize_samples_per_example,
                    dim=0,
                ),
                torch.split(
                    z[:, :2],
                    split_size_or_sections=args.num_visualize_samples_per_example,
                    dim=0,
                ),
                torch.split(
                    z_prior,
                    split_size_or_sections=args.num_visualize_samples_per_example,
                    dim=0,
                ),
            )
        ):
            assert (
                _z0.shape
                == _z.shape
                == _z_prior.shape
                == (args.num_visualize_samples_per_example, 2)
            ), f"The shape of _z0, _z, _z_prior for plotting should be ({args.num_visualize_samples_per_example}, 2)."

            figure = visualize_latent_and_prior(
                Zs=[Z.detach().cpu().numpy() for Z in [_z0, _z, _z_prior]],
                labels=[r"$Z_0$", r"$Z_K$", r"prior $N(0, I)$"],
                title=f"Example{i+1}",
            )
            figures.append(figure)
        return figures
