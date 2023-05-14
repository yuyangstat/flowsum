import datasets
import torch
from datasets import load_dataset, concatenate_datasets

from collections import Counter
from functools import partial


# not only the original datasets will be cached, but also the processed data,
# therefore, everytime that I modify the processed data format, I need to remove the
# cached processed files
def get_processed_data(
    data_name,
    split,
    tokenizer,
    batch_size,
    encoder_max_length=512,
    decoder_max_length=128,
    select_indices=None,
):
    assert split in ["train", "validation", "test"]
    if data_name == "cnndm":
        data = datasets.load_dataset("cnn_dailymail", "3.0.0", split=split)
        if select_indices is not None:
            data = data.select(select_indices)
        data = data.map(
            partial(
                process_data_to_model_inputs,
                tokenizer=tokenizer,
                encoder_max_length=encoder_max_length,
                decoder_max_length=decoder_max_length,
            ),
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "highlights", "id"],
        )
        data.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "decoder_input_ids",
                "decoder_attention_mask",
                "labels",
                "decoder_bows",
            ],
        )
    return data


def process_data_to_model_inputs(
    batch, tokenizer, encoder_max_length, decoder_max_length
):
    """
    The reason to name bows as "decoder_bows" is:
        _prepare_encoder_decoder_kwargs_for_generation() in generate() treats all
        kwargs not starting with ['decoder', 'cross_attn', 'use_cache'] as the encoder
        arguments and pass to the encoder.
    """
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch["highlights"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    # get normalized bows
    batch["decoder_bows"] = get_normalized_bows(
        input=batch["article"], tokenizer=tokenizer
    )

    return batch


def get_normalized_bows(input, tokenizer):
    _input_ids = tokenizer(input, truncation=False).input_ids
    _input_counters = [dict(Counter(_input)) for _input in _input_ids]
    _input_lens = [len(_input) for _input in _input_ids]
    return torch.FloatTensor(
        [
            [cnt.get(i, 0) / _input_len for i in range(tokenizer.vocab_size)]
            for cnt, _input_len in zip(_input_counters, _input_lens)
        ]  # need to be float since "addmm_cuda" not implemented for 'Long'
    )


def check_cnndm():
    """Check the information of the CNN/Daily Mail dataset."""
    for split in ["train", "validation", "test"]:
        print(f"\n*======== {split.capitalize()} ========*\n")
        data = datasets.load_dataset("cnn_dailymail", "3.0.0", split=split)
        print(data)


def preprocess_data_function(
    examples,
    text_column,
    summary_column,
    prefix,
    tokenizer,
    max_source_length,
    max_target_length,
    padding,
    ignore_pad_token_for_loss,
    add_bows,
):
    """
    Notes:
        For both EncoderDecoderModel and *ForConditionalGeneration models, "decoder_input_ids" and
            "decoder_attention_mask" are needed.
        In the model level, according to https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L1216,
            BART will automatically create "decoder_input_ids" if it is not provided and BartDecoder will
            generate "decoder_attention_mask" as in https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L1043.
            The same for PEGASUS. For T5, the coding logic is slightly different, but both features will
            also be generated if None. As for EncoderDecoderModel, the situation is a bit different in that
            "decoder_input_ids" will not be automatically generated inside the model if None but
            "decoder_attention_mask" will be automatically generated inside the decoder if None.
        If we instantiate the trainer without using data_collator from DataCollatorForSeq2Seq, then for the
            EncoderDecoderModel class, we need to explicitly provide "decoder_input_ids" and
            "decoder_attention_mask". This is what https://colab.research.google.com/drive/1Ekd5pUeCX7VOrMx94_czTkwNtLN32Uyu?usp=sharing
            did in its data processing function.
        If, however, we instantiate the trainer with a data_collator, then when called, the data_collator
            will run `model.prepare_decoder_input_ids_from_labels()` to generate the "decoder_input_ids"
            automatically. The problem we then need to solve is to make sure the
            `prepare_decoder_input_ids_from_labels()` function is correctly generating the needed "decoder_input_ids".
            For BART, Pegasus, and EncoderDecoderModel class, they all run `shift_tokens_right()` on the labels
            to generate the decoder_input_ids. This works in general. However, as mentioned in https://colab.research.google.com/drive/1Ekd5pUeCX7VOrMx94_czTkwNtLN32Uyu?usp=sharing,
            BERT automatically shifts the labels, so if the EncoderDecoderModel has BERT as its components,
            the labels should correspond exactly to `decoder_input_ids`. Therefore, for NFBert2Bert,
            we need to modify the `prepare_decoder_input_ids_from_labels()` to not shift the tokens to the right.
    """
    # remove pairs where at least one record is None
    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[summary_column][i]:
            inputs.append(examples[text_column][i])
            targets.append(examples[summary_column][i])

    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=padding,
        truncation=True,
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]

    # get normalized bows
    if add_bows:
        model_inputs["decoder_bows"] = get_normalized_bows(
            input=inputs, tokenizer=tokenizer
        )

    return model_inputs


SUMMARIZATION_NAME_MAPPING = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
    "bart_cnndm": ("article", "highlights"),
    "distil_cnndm": ("article", "highlights"),
    "distil_xsum": ("document", "summary"),
    "newsroom": ("text", "summary"),
    "scientific_papers": ("article", "abstract"),
}


def get_text_summary_column(data_args, column_names):
    # Get the column names for input/target.
    dataset_columns = SUMMARIZATION_NAME_MAPPING.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = (
            dataset_columns[0] if dataset_columns is not None else column_names[0]
        )
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = (
            dataset_columns[1] if dataset_columns is not None else column_names[1]
        )
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )
    return text_column, summary_column


DISTILLED_DATA_NAME_MAPPING = {
    "cnn_dailymail": "yuyang/distil_cnndm",  # the (relative/absolute) path to the data loading directory
    "bart_cnndm": "yuyang/distil_cnndm",
    "xsum": "yuyang/distil_xsum",
}


def get_raw_datasets(data_args, model_args):
    """Get raw textual data.

    Note:
        (1) If data_args.augment_distilled is True, we only augment
            the training data, since the validation and test of the
            distilled version are exactly the same as the original
            version.
        (2) The current script requires the distilled data to be loaded from
            Hugging Face Hub, which contains the prefix "yuyang/". If users
            load the distilled data from a local directory with the data
            loading script, then DISTILLED_DATA_NAME_MAPPING needs to be
            modified accordingly, namely, remove the prefix "yuyang/".
        (3) Updates on 03/14: switch the order of distilled and original data. The logic
            is to have the model learn more from the original data in the end. The
            empirical performance is not tested yet.
    """
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        if data_args.dataset_name == "newsroom":
            raw_datasets = load_dataset(
                "newsroom", data_dir="~/.cache/huggingface/datasets/newsroom"
            )
        else:
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,  # 'csv' or 'json'
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # handle distilled data if needed
    assert not (
        data_args.augment_distilled and data_args.distilled_only
    ), "`augment_distilled` and `distilled_only` cannot both be True."

    if data_args.augment_distilled or data_args.distilled_only:
        distilled_dataset = load_dataset(
            DISTILLED_DATA_NAME_MAPPING[data_args.dataset_name],
            data_args.distilled_dataset_config_name
            if data_args.distilled_dataset_config_name is not None
            else data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if data_args.augment_distilled:
            raw_datasets["train"] = concatenate_datasets(
                [distilled_dataset["train"], raw_datasets["train"]]
            )
        else:
            raw_datasets["train"] = distilled_dataset["train"]

    return raw_datasets


def get_test_dataset_and_text_summary_column(dataset_name):
    """This function is used for exploratory analysis in ipython notebooks."""
    if dataset_name in ["cnn_dailymail", "bart_cnndm"]:
        raw_datasets = load_dataset(dataset_name, "3.0.0")
    elif dataset_name in ["arxiv", "pubmed"]:
        raw_datasets = load_dataset("scientific_papers", dataset_name)
    else:
        raw_datasets = load_dataset(dataset_name)

    if dataset_name in ["arxiv", "pubmed"]:
        text_column, summary_column = SUMMARIZATION_NAME_MAPPING.get(
            "scientific_papers", None
        )
    else:
        text_column, summary_column = SUMMARIZATION_NAME_MAPPING.get(dataset_name, None)
    dataset = raw_datasets["test"]
    return dataset, text_column, summary_column
