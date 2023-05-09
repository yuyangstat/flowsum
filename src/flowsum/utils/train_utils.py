import torch
from dataclasses import dataclass, field
from typing import Optional, Union
import evaluate

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    BartForConditionalGeneration,
)
from transformers.trainer_utils import IntervalStrategy

from flowsum.models.nfbert2bert import NFBert2Bert
from flowsum.models.nfpegasus import NFPegasus
from flowsum.models.nfbart import NFBart
from flowsum.models.nft5 import NFT5
from flowsum.models.nfgpt2 import NFGPT2
from flowsum.trainers.trainer_bows_seq2seq import BOWsSeq2SeqTrainer
from flowsum.trainers.trainer_time import TimeSeq2SeqTrainer
from flowsum.trainers.trainer_lag import LagBOWsSeq2SeqTrainer


# load rouge for validation
rouge = evaluate.load("rouge")


def compute_metrics(pred, tokenizer):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str)

    return {k: round(v, 4) for k, v in rouge_output.items()}


CUSTOM_MODELS = [
    "nf-bert2bert",
    "nf-pegasus",
    "nf-bart",
    "nf-t5",
    "nf-gpt2",
    "nf-bert2bert-pretrained",
    "nf-bert2bert-finetuned",
    "nf-pegasus-finetuned",
    "nf-pegasus-finetuned-xsum",
    "nf-pegasus-finetuned-multinews",
    "nf-pegasus-finetuned-arxiv",
    "nf-pegasus-finetuned-pubmed",
    "nf-pegasus-finetuned-wikihow",
    "nf-pegasus-finetuned-aeslc",
    "nf-pegasus-finetuned-gigaword",
    "nf-pegasus-finetuned-newsroom",
    "nf-pegasus-finetuned-reddit_tifu",
    "nf-bart-finetuned",
    "nf-distilbart",  # cnn-6-6
    "nf-distilbart_12_3",
    "nf-distilbart_12_6",
    "nf-distilbart_6_6-xsum",
    "nf-distilbart_12_3-xsum",
    "nf-distilbart_12_6-xsum",
    "nf-bart-finetuned-xsum",
    "nf-bart-finetuned-multinews",
    "nf-bart-finetuned-samsum",
    "nf-bart-finetuned-science",
    "nf-bart-finetuned-newsroom",
    "nf-t5-pretrained",
    "nf-gpt2-pretrained",
    "bart-original",
]


def load_custom_model(model_name, config, tokenizer, model_args, model_ckpt_path=None):
    """
    Current pretrained ckpts are fixed. May generalize it if needed.

    Args:
        model_args: should be an instance of ModelArguments.

    """
    nf_args = model_args.nf_args

    assert (
        model_name in CUSTOM_MODELS
    ), f"The model_name should be one of {CUSTOM_MODELS}."

    if model_name == "nf-bert2bert":
        model = NFBert2Bert(config=config, **nf_args)
    elif model_name == "nf-pegasus":
        model = NFPegasus(config=config, **nf_args)
    elif model_name == "nf-bart":
        model = NFBart(config=config, **nf_args)
    elif model_name == "nf-t5":
        model = NFT5(config=config, **nf_args)
    elif model_name == "nf-gpt2":
        model = NFGPT2(config=config, **nf_args)
    elif model_name == "nf-bert2bert-pretrained":
        model = NFBert2Bert.from_encoder_decoder_pretrained(
            "bert-base-uncased", "bert-base-uncased", **nf_args
        )
    elif model_name == "nf-bert2bert-finetuned":
        model = NFBert2Bert.from_pretrained(
            "patrickvonplaten/bert2bert_cnn_daily_mail", **nf_args
        )
    elif model_name == "nf-pegasus-finetuned":
        model = NFPegasus.from_pretrained("google/pegasus-cnn_dailymail", **nf_args)
    elif model_name == "nf-pegasus-finetuned-xsum":
        model = NFPegasus.from_pretrained("google/pegasus-xsum", **nf_args)
    elif model_name == "nf-pegasus-finetuned-multinews":
        model = NFPegasus.from_pretrained("google/pegasus-multi_news", **nf_args)
    elif model_name == "nf-pegasus-finetuned-arxiv":
        model = NFPegasus.from_pretrained("google/pegasus-arxiv", **nf_args)
    elif model_name == "nf-pegasus-finetuned-pubmed":
        model = NFPegasus.from_pretrained("google/pegasus-pubmed", **nf_args)
    elif model_name == "nf-pegasus-finetuned-wikihow":
        model = NFPegasus.from_pretrained("google/pegasus-wikihow", **nf_args)
    elif model_name == "nf-pegasus-finetuned-aeslc":
        model = NFPegasus.from_pretrained("google/pegasus-aeslc", **nf_args)
    elif model_name == "nf-pegasus-finetuned-gigaword":
        model = NFPegasus.from_pretrained("google/pegasus-gigaword", **nf_args)
    elif model_name == "nf-pegasus-finetuned-newsroom":
        model = NFPegasus.from_pretrained("google/pegasus-newsroom", **nf_args)
    elif model_name == "nf-pegasus-finetuned-reddit_tifu":
        model = NFPegasus.from_pretrained("google/pegasus-reddit_tifu", **nf_args)
    elif model_name == "nf-bart-finetuned":
        model = NFBart.from_pretrained("facebook/bart-large-cnn", **nf_args)
    elif model_name == "nf-distilbart":
        model = NFBart.from_pretrained("sshleifer/distilbart-cnn-6-6", **nf_args)
    elif model_name == "nf-distilbart_12_3":
        model = NFBart.from_pretrained("sshleifer/distilbart-cnn-12-3", **nf_args)
    elif model_name == "nf-distilbart_12_6":
        model = NFBart.from_pretrained("sshleifer/distilbart-cnn-12-6", **nf_args)
    elif model_name == "nf-distilbart_6_6-xsum":
        model = NFBart.from_pretrained("sshleifer/distilbart-xsum-6-6", **nf_args)
    elif model_name == "nf-distilbart_12_3-xsum":
        model = NFBart.from_pretrained("sshleifer/distilbart-xsum-12-3", **nf_args)
    elif model_name == "nf-distilbart_12_6-xsum":
        model = NFBart.from_pretrained("sshleifer/distilbart-xsum-12-6", **nf_args)
    elif model_name == "nf-bart-finetuned-xsum":
        model = NFBart.from_pretrained("facebook/bart-large-xsum", **nf_args)
    elif model_name == "nf-bart-finetuned-multinews":
        model = NFBart.from_pretrained(
            "nikhedward/bart-large-cnn-finetuned-multi-news1", **nf_args
        )
    elif model_name == "nf-bart-finetuned-samsum":
        model = NFBart.from_pretrained("knkarthick/MEETING_SUMMARY", **nf_args)
    elif model_name == "nf-bart-finetuned-science":
        model = NFBart.from_pretrained("theojolliffe/bart-cnn-science", **nf_args)
    elif model_name == "nf-bart-finetuned-newsroom":
        model = NFBart.from_pretrained(
            "yuyang/bart-large-cnn-finetuned-newsroom", **nf_args
        )
    elif model_name == "nf-t5-pretrained":
        model = NFT5.from_pretrained("t5-base", **nf_args)
    elif model_name == "nf-gpt2-pretrained":
        model = NFGPT2.from_pretrained("gpt2", **nf_args)
    elif model_name == "bart-original":
        model = BartForConditionalGeneration(config=config)

    if model_name in ["nf-bert2bert", "nf-bert2bert-pretrained"]:
        assert tokenizer.name_or_path == "bert-base-uncased"
        # set special tokens
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        # sensible parameters for beam search
        model.config.vocab_size = model.config.decoder.vocab_size
        beam_search_args = model_args.beam_search_args
        model.config.max_length = beam_search_args["max_length"]
        model.config.min_length = beam_search_args["min_length"]
        model.config.no_repeat_ngram_size = beam_search_args["no_repeat_ngram_size"]
        model.config.early_stopping = beam_search_args["early_stopping"]
        model.config.length_penalty = beam_search_args["length_penalty"]
        model.config.num_beams = beam_search_args["num_beams"]

    if model_name in ["nf-bart", "nf-bart-finetuned-xsum"]:
        beam_search_args = model_args.beam_search_args
        if beam_search_args["length_penalty"] is not None:
            model.config.length_penalty = beam_search_args["length_penalty"]

    if model_ckpt_path is not None:
        # [TODO] examine the device bug
        state_dict = torch.load(model_ckpt_path)
        model.load_state_dict(state_dict)

    return model


SUPPORTED_TRAINERS = [
    "seq2seq",
    "time_seq2seq",
    "bows_seq2seq",
    "lag_bows_seq2seq",
    "lag_bows_seq2seq_v2",
]


def get_trainer_cls(trainer_name: str):
    """Get trainer's class from name."""
    assert (
        trainer_name in SUPPORTED_TRAINERS
    ), f"We currently only support {SUPPORTED_TRAINERS}."
    if trainer_name == "seq2seq":
        return Seq2SeqTrainer
    elif trainer_name == "bows_seq2seq":
        return BOWsSeq2SeqTrainer
    elif trainer_name == "lag_bows_seq2seq":
        return LagBOWsSeq2SeqTrainer
    elif trainer_name == "time_seq2seq":
        return TimeSeq2SeqTrainer


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )
    nf_args: dict = field(
        default_factory=lambda: {
            "q_input_type": None,
            "q_hidden_dims": None,
            "q_act": None,
            "q_dropout": None,
            "nf_name": None,
            "nf_latent_size": None,
            "nf_num_layers": None,
            "nf_loss_weight": 1.0,
            "beta_vae_constant": 0.0,
            "output_kld": False,
        },
        metadata={
            "help": (
                "The arguments for normalizing flows."
                "q_input_type: should be either 'avg_embed' or 'bows'. "
                "q_hidden_dims: a list of integers, specifies the q_net. q_net will have len(q_hidden_dims) hidden layers. "
            )
        },
    )
    beam_search_args: dict = field(
        default_factory=lambda: {
            "max_length": None,
            "min_length": None,
            "no_repeat_ngram_size": None,
            "early_stopping": None,
            "length_penalty": None,
            "num_beams": None,
        },
        metadata={
            "help": (
                "The arguments for beam search."
                "Only used when the model is `nf-bert2bert` or `nf-bert2bert-pretrained`."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    add_bows: bool = field(
        default=False,
        metadata={"help": "Whether to add normalized bag of words to data."},
    )
    augment_distilled: bool = field(
        default=False,
        metadata={
            "help": "Whether to augment the training data with the distilled data."
        },
    )
    distilled_dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the distilled dataset to use (via the datasets library)."
        },
    )
    distilled_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to train on the distilled data only. If True, the training data will not include the original data."
                "`augment_distilled` and `distilled_only` cannot both be True."
            )
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class MiscArguments:
    """
    Miscellaneous arguments.
    """

    trainer_name: str = field(
        default="bows_seq2seq",
        metadata={
            "help": (
                "The trainer's name."
                "Should be one of ['seq2seq', 'bows_seq2seq', 'lag_bows_seq2seq']."
            )
        },
    )
    early_stopping_patience: int = field(
        default=1,
        metadata={
            "help": (
                "Stop training when the specified metric worsens for this number of evaluation calls."
                "Used in EarlyStoppingCallback."
                "Need to be set as a larger value if training_args.eval_steps is too small due to the high variance."
            )
        },
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "How much the specified metric must improve to satisfy early stopping conditions. "
                "Used in EarlyStoppingCallback."
            )
        },
    )
    generate_pseudo_labels: bool = field(
        default=False,
        metadata={
            "help": "Whether to generate pseudo labels. If true, will make predictions on the training data."
        },
    )


@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """Training arguments used for lag training and visualization.

    This way of writing depends on that the origianl TrainingArguments has all its field with default values,
        since we cannot have default arguments following non-default arguments.

    References:
        (1) https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
    """

    num_alternate_epochs: float = field(
        default=1.0,
        metadata={
            "help": "The number of epochs to train with aggressive (alternative) training at the beginning."
        },
    )
    num_variational_iter_per_alter: int = field(
        default=20,
        metadata={
            "help": (
                "The number of variational updates for every round of alternation. ",
                "For every alternation, there will be (num_variational_iter_per_alter - 1) variational params update ",
                "and 1 model params update.",
            )
        },
    )
    visualize_latent_dist: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to visualize the latent distribution during training. "
                "If True, VisualTensorBoardCallback will added and the default TensorBoardCallback "
                "will be removed. "
                "The visualization includes scatterplots and kernel density plots of z_0, z_K, and N(0, I). "
                "All the generated images will be logged down in Tensorboard. "
            )
        },
    )
    visualization_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The visualization strategy to use."},
    )
    visual_steps: Optional[int] = field(
        default=1000, metadata={"help": "Run an visualization every X steps."}
    )
    num_visualize_samples_per_example: int = field(
        default=300,
        metadata={
            "help": "Number of samples to plot for each distribution and each example in visualization. "
        },
    )
    num_visualize_examples: int = field(
        default=4,
        metadata={
            "help": (
                "Number of examples to plot to Tensorboard . ",
                "If greater than the training batch size, then only batch_size examples will be visualized. ",
            )
        },
    )
