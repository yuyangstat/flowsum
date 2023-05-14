"""
This file intends to implement the CAAT algorithm.

The trainer class here inherits from the BOWsSeq2SeqTrainer() in trainer_bows_seq2seq.py 
and we focus on modifying the inner training loop here.

References: 
(1) https://github.com/jxhe/vae-lagging-encoder/blob/master/text.py
(2) https://arxiv.org/abs/1901.05534
"""

import math
import sys
import os
import time
import shutil
import random
import warnings
import numpy as np

from tqdm.auto import tqdm

# Integrations must be imported before ML frameworks:
from transformers.integrations import hp_params

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.trainer_utils import (
    TrainOutput,
    HPSearchBackend,
    has_length,
    speed_metrics,
    PREFIX_CHECKPOINT_DIR,
)
from transformers.utils import (
    logging,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    get_parameter_names,
    reissue_pt_warnings,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer_callback import TrainerState
from transformers.pytorch_utils import is_torch_less_than_1_11, ALL_LAYERNORM_LAYERS
from transformers.trainer import Trainer
from transformers.optimization import get_scheduler


from flowsum.trainers.trainer_bows_seq2seq import BOWsSeq2SeqTrainer

logger = logging.get_logger(__name__)

TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class CAATBOWsSeq2SeqTrainer(BOWsSeq2SeqTrainer):
    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,  # trainer.args will be passed to it
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        """Modified version of BOWsSeq2SeqTrainer._inner_training_loop().

        The main idea is to enable inside loopped training for variational parameters.

        Note:
            (1) For coding simplicity, I removed all options related to
                - is_sagemaker_mp_enabled()
                - is_torch_tpu_available()
                - is_apex_available()
                - self.sharded_ddp (use it as None)
                - args.deepspeed (use it as None), this could be added if we have time.
            (2) There are two strategies to deal with the examples for the inner and the outer loop.
                (1) The outer loop will iterate over all examples for each epoch, and for each step
                    inside the epoch, the inner loop will first iterate over a subset of examples
                    and then pass on to the outer loop. This strategy is time-consuming.
                (2) For the first few epochs, an example would either be processed by the inner loop
                    or by the outer loop. During this stage, both the inner and the outer loop will
                    not see all data in one epoch, but only partial. And the ratio of examples they
                    can see is (args.num_variational_iter_per_alter - 1): 1. After the first few epochs, we update the model
                    parameters and the variational parameters together. In this stage, the two
                    optimizers will be able to see all data. This strategy should be more time
                    efficient. Anothe benefit is that we can now monitor the loss in the inner
                    loop and can get rid of the potential iterator problem inside the inner loop.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Set up the example used for visualization
        #   in order to track the changes, we need to fix the example.
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

        # for the ease of coding, we only consider the case where has_length(train_dataloader) is True
        self.create_optimizer_and_scheduler(
            num_training_steps=max_steps,
            num_alternate_training_steps=math.ceil(
                max_steps / args.num_train_epochs * args.num_alternate_epochs
            ),
        )

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

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
        self.callback_handler.optimizer = self.optimizer  # [TODO] monitor the effect
        self.callback_handler.lr_scheduler = self.lr_scheduler  # [TODO] monitor
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

            # Whether to do alternate training
            aggressive_flag = epoch < args.num_alternate_epochs
            logger.info(
                f"Train with {'' if aggressive_flag else 'non-'}aggressive training."
            )

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
                    ) = self.training_step(
                        model, inputs
                    )  # loss.backward() is performed inside self.training_step()

                if args.logging_nan_inf_filter and (
                    torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)
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

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    if aggressive_flag:
                        # Alternatively update the variational optimizer and the model optimizer
                        if (
                            step + 1
                        ) % args.num_variational_iter_per_alter == 0:  # run model optimizer
                            # Gradient clipping for the model optimizer
                            if (
                                args.max_grad_norm is not None
                                and args.max_grad_norm > 0
                            ):
                                if self.do_grad_scaling:
                                    self.scaler.unscale_(self.model_optimizer)
                                if hasattr(self.model_optimizer, "clip_grad_norm"):
                                    # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                    self.model_optimizer.clip_grad_norm(
                                        args.max_grad_norm
                                    )
                                elif hasattr(model, "clip_grad_norm_"):
                                    # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                    model.clip_grad_norm_(args.max_grad_norm)
                                else:
                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    nn.utils.clip_grad_norm_(
                                        model.parameters(),  # clipping extra gradients doesn't matter, so we don't modify it to be model part parameters specifically
                                        args.max_grad_norm,
                                    )
                            # Optimizer step for the model optimizer
                            optimizer_was_run = True
                            if self.do_grad_scaling:
                                scale_before = self.scaler.get_scale()
                                self.scaler.step(self.model_optimizer)
                                self.scaler.update()
                                scale_after = self.scaler.get_scale()
                                optimizer_was_run = scale_before <= scale_after
                            else:
                                self.model_optimizer.step()

                            if optimizer_was_run:
                                self.model_lr_scheduler.step()
                        else:
                            # Gradient clipping for the variational optimizer
                            if (
                                args.max_grad_norm is not None
                                and args.max_grad_norm > 0
                            ):
                                if self.do_grad_scaling:
                                    self.scaler.unscale_(self.variational_optimizer)
                                if hasattr(
                                    self.variational_optimizer, "clip_grad_norm"
                                ):
                                    self.variational_optimizer.clip_grad_norm(
                                        args.max_grad_norm
                                    )
                                elif hasattr(model, "clip_grad_norm_"):
                                    model.clip_grad_norm_(args.max_grad_norm)
                                else:
                                    nn.utils.clip_grad_norm_(
                                        model.parameters(),  # clipping extra gradients doesn't matter, so we don't modify it to be variational parameters specifically
                                        args.max_grad_norm,
                                    )
                            # Optimizer step for the variational optimizer
                            optimizer_was_run = True
                            if (
                                self.do_grad_scaling
                            ):  # [TODO] need to double check the self.scaler.
                                scale_before = self.scaler.get_scale()
                                self.scaler.step(self.variational_optimizer)
                                self.scaler.update()
                                scale_after = self.scaler.get_scale()
                                optimizer_was_run = scale_before <= scale_after
                            else:
                                self.variational_optimizer.step()

                            if optimizer_was_run:
                                self.variational_lr_scheduler.step()
                    else:  # Standard Training using self.optimizer
                        # Gradient clipping for the optimizer
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if self.do_grad_scaling:
                                self.scaler.unscale_(self.optimizer)
                            if hasattr(self.optimizer, "clip_grad_norm"):
                                # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                                self.optimizer.clip_grad_norm(args.max_grad_norm)
                            elif hasattr(model, "clip_grad_norm_"):
                                # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                                model.clip_grad_norm_(args.max_grad_norm)
                            else:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    model.parameters(),  # clipping extra gradients doesn't matter, so we don't modify it to be model part parameters specifically
                                    args.max_grad_norm,
                                )
                        # Optimizer step for the optimizer
                        optimizer_was_run = True
                        if self.do_grad_scaling:
                            scale_before = self.scaler.get_scale()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            scale_after = self.scaler.get_scale()
                            optimizer_was_run = scale_before <= scale_after
                        else:
                            self.optimizer.step()

                        if optimizer_was_run:
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
            if args.local_rank != -1:
                dist.barrier()

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

    def create_optimizer_and_scheduler(
        self, num_training_steps: int, num_alternate_training_steps: int
    ):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.

        Args:
            num_training_steps (int): The total number of training steps to do.
            num_alternate_training_steps (int): The number of alternate training steps.
        """
        self.create_optimizer()
        self.create_scheduler(
            num_training_steps=num_training_steps,
            num_alternate_training_steps=num_alternate_training_steps,
            optimizer=self.optimizer,
            model_optimizer=self.model_optimizer,
            variational_optimizer=self.variational_optimizer,
        )

    def create_optimizer(self):
        """Setup the optimizer.

        Modified version of Trainer.create_optimizer().

        The main idea is to have three separate optimizers: one for variational parameters, one
            for encoder-decoder model parameters, and the other for all parameters.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.

        Note:
            (1) For the sake of coding efficiency, we consider self.sharded_ddp != ShardedDDPOption.SIMPLE.
            (2) Despite of the value of self.optimizer, we will create model and variational optimizer anyway.
        """
        opt_model = self.model
        nf_latent_parameters = get_parameter_names(opt_model.nf_latent_model, [])

        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        # Currently, we use the same type of optimizer for the model_optimizer and variational_optimizer.
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )

        # set up model optimizer
        model_optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n in decay_parameters and n not in nf_latent_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if n not in decay_parameters and n not in nf_latent_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        variational_optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.nf_latent_model.named_parameters()
                    if n in decay_parameters
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.nf_latent_model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        self.model_optimizer = optimizer_cls(
            model_optimizer_grouped_parameters, **optimizer_kwargs
        )
        self.variational_optimizer = optimizer_cls(
            variational_optimizer_grouped_parameters, **optimizer_kwargs
        )
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n in decay_parameters
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if n not in decay_parameters
                    ],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum(
                        dict(
                            (p.data_ptr(), p.numel()) for p in module.parameters()
                        ).values()
                    )
                    print(f"skipped {module}: {skipped/2**20}M params")
                    manager.register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            print(f"skipped: {skipped/2**20}M params")

        return self.optimizer, self.model_optimizer, self.variational_optimizer

    def create_scheduler(
        self,
        num_training_steps: int,
        num_alternate_training_steps: int,
        optimizer: torch.optim.Optimizer,
        model_optimizer: torch.optim.Optimizer,
        variational_optimizer: torch.optim.Optimizer,
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Modified version of Trainer.create_scheduler().

        The main idea is to have two separate lr schedulers for the two optimizers:
            one for the variational optimizer, the other for the model optimizer.

        Args:
            num_training_steps (int): The total number of training steps to do.
            num_alternate_training_steps (int): The number of alternate training steps.

        Note:
            (1) Despite of the value of self.lr_scheduler, we will create model and variational lr scheduler anyway.
            (2) We don't set the num_warmup_steps and num_training_steps for the
                variational_lr_scheduler, since we don't know the number of steps inside each loop.
        """
        self.model_lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=model_optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=math.ceil(
                num_alternate_training_steps / self.args.num_variational_iter_per_alter
            ),
        )
        self.variational_lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=variational_optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=math.ceil(
                num_alternate_training_steps
                / self.args.num_variational_iter_per_alter
                * (self.args.num_variational_iter_per_alter - 1)
            ),
        )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps - num_alternate_training_steps,
            )

        return self.lr_scheduler, self.model_lr_scheduler, self.variational_lr_scheduler

    def _get_learning_rate(self):
        """
        Modified version of transformers.trainer_pt_utils._get_learning_rate().

        This function will be called inside _maybe_log_save_evaluate() for logging purposes.
            We consider logging down the lr_scheduler's learning rate only, so for the first
            few epochs when aggressive training is applied, it will remain unchanged.
        """
        last_lr = self.lr_scheduler.get_last_lr()[0]
        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
        return last_lr

    def _save_checkpoint(self, model, trial, metrics=None):
        """Modified version of trainer._save_checkpoint().

        The model_ and variational_ optimizer and lr_scheduler are saved separately.
            [TODO] may think about saving them jointly.

        Note:
            (1) All code relating to tpu, deepspeed, shared_ddp, are removed for simplicity.
        """
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        # Save optimizer and scheduler
        if self.args.should_save:
            # deepspeed.save_checkpoint above saves model/optim/sched
            if self.optimizer.state_dict() is not None:
                torch.save(
                    self.optimizer.state_dict(),
                    os.path.join(output_dir, OPTIMIZER_NAME),
                )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(
                        self.lr_scheduler.state_dict(),
                        os.path.join(output_dir, SCHEDULER_NAME),
                    )
            else:
                torch.save(
                    self.model_optimizer.state_dict(),
                    os.path.join(output_dir, f"model_{OPTIMIZER_NAME}"),
                )
                torch.save(
                    self.variational_optimizer.state_dict(),
                    os.path.join(output_dir, f"variational_{OPTIMIZER_NAME}"),
                )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(
                        self.model_lr_scheduler.state_dict(),
                        os.path.join(output_dir, f"model_{SCHEDULER_NAME}"),
                    )
                    torch.save(
                        self.variational_lr_scheduler.state_dict(),
                        os.path.join(output_dir, f"variational_{SCHEDULER_NAME}"),
                    )
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(
                    self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME)
                )

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(
                rng_states,
                os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"),
            )

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
