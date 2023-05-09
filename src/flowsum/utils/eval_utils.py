import nltk
import numpy as np
import datasets
import evaluate
from collections import Counter
from typing import List, Union

from flowsum.utils.data_utils import get_normalized_bows


def generate_summary(batch, model, tokenizer, device="cuda:4"):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    # get normalized bows
    decoder_bows = get_normalized_bows(input=batch["article"], tokenizer=tokenizer).to(
        device
    )

    outputs = model.generate(
        input_ids, attention_mask=attention_mask, decoder_bows=decoder_bows
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


def postprocess_text(preds, labels):
    """
    (1) PegasusTokenizer generates '<n>' at the beginning of sentences, if not dealt with, the resulting ROUGE scores will be lower.
        Therefore, we replace '<n>' with '\n' as suggested in https://github.com/huggingface/transformers/issues/6844#issuecomment-696885044.
    """
    preds = [pred.strip().replace("<n>", "\n") for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds, tokenizer, metric, ignore_pad_token_for_loss):
    """
    Note:
        This function is used inside trainer and both the preds and labels are
        truncated up to data_args.val_max_target_length. We can use it for
        evaluation during training, but not for the final ROUGE evaluation.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def evaluate_rouge_on_file(
    prediction_path,
    dataset=None,
    dataset_name="cnn_dailymail",
    dataset_config_name="3.0.0",
    split="test",
    text_column="article",
    summary_column="highlights",
):
    """
    This function is used to evaluate the generated summaries in terms of ROUGE scores. Unlike
        compute_metrics() used in trainer, the evaluation here is based on
        the full generated text. After our experiments, we found that the
        datasets package gives much smaller metric values than the evaluate
        package and the calculate_rouge() in https://github.com/yuyangstatistics/transformers/blob/main/examples/legacy/seq2seq/utils.py#L508,
        and the latter two gives very similar results. Therefore, we use evaluate here.

    Note:
        It is important to specify `use_stemmer = True`, otherwise, the results
            will be much smaller. And only when `use_stemmer = True` will we have
            similar results as calculate_rouge().
    """
    with open(prediction_path, "r") as fp:
        preds = fp.read().split("\n")
    if dataset is None:
        dataset = datasets.load_dataset(dataset_name, dataset_config_name, split=split)
    documents = dataset[text_column]
    labels = dataset[summary_column]
    labels = [
        l for d, l in zip(documents, labels) if d and l
    ]  # only keep the non-empty ones
    preds, labels = postprocess_text(preds, labels)
    metric = evaluate.load("rouge")
    result = metric.compute(predictions=preds, references=labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    return result


def evaluate_repetition_on_file(
    prediction_path: str, bandwidth: int = 16, n: Union[int, List[int]] = [1, 2, 3, 4]
):
    """
    This function is used to evaluate the generated summaries in terms of repetition.

    The smaller the scores are, the less repetitive the text is.
    """
    with open(prediction_path, "r") as fp:
        preds = fp.read().split("\n")
    preds = [pred.strip().replace("<n>", "\n") for pred in preds]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

    if not isinstance(n, list):
        n = [n]
    rep_scores = [get_rep_scores(pred, bandwidth=bandwidth, n=n) for pred in preds]
    result = dict(
        zip(
            ["rep_w", "rep_r"] + [f"rep_seq_{i}" for i in n],
            list(np.array(rep_scores).mean(axis=0)),
        )
    )
    result = {k: round(v * 100, 4) for k, v in result.items()}
    return result


def get_rep_scores(
    sent: str, bandwidth: int = 16, n: Union[int, List[int]] = [1, 2, 3, 4]
) -> List[float]:
    """Get repetition scores.

    Reference: https://github.com/fuzihaofzh/repetition-problem-nlg/blob/main/src/eval_metrics.py

    Output order: ["rep_w", "rep_r"] + [f"rep_seq_{i}" for i in n]
    """
    if not isinstance(n, list):
        n = [n]
    sent = sent.split()
    rep_scores = [
        get_word_rep_score(sent, bandwidth),
        get_rep_r_score(sent),
    ] + [get_seq_rep_n_score(sent, i) for i in n]
    return rep_scores


def get_word_rep_score(sent: List[str], bandwidth=16):
    cnt = 0
    for i, w in enumerate(sent):
        if w in sent[max(i - bandwidth, 0) : i]:
            cnt += 1
    return cnt / len(sent)


def get_seq_rep_n_score(sent: List[str], n=3):
    ngrams = []
    if len(sent) <= n:
        return 0.0
    else:
        for i in range(len(sent) - n + 1):
            ngrams.append(tuple(sent[i : i + n]))
        return 1.0 - len(set(ngrams)) / len(ngrams)


def get_rep_r_score(sent: List[str]):
    if len(sent) < 2:
        return 0.0
    counter = Counter([f"{sent[i]} {sent[i+1]}" for i in range(len(sent) - 1)])
    label = [0] * len(sent)
    for i in range(len(sent) - 1):
        if counter[f"{sent[i]} {sent[i+1]}"] > 1:
            label[i] = label[i + 1] = 1
    return sum(label) / len(label)


def evaluate_bertscore_on_file(
    prediction_path,
    dataset=None,
    dataset_name="cnn_dailymail",
    dataset_config_name="3.0.0",
    split="test",
    text_column="article",
    summary_column="highlights",
):
    """
    This function is used to evaluate the generated summaries in terms of BERTScore.

    Notes:
        (1) The default model for lang="en" is Roberta-Large.
        (2) Unlike the rouge and repetition measures, this measure requires running the BERT or
            Roberta model as specified, so we need to be mindful of the CUDA space when running
            this evaluation.

    Reference:
        (1) https://huggingface.co/spaces/evaluate-metric/bertscore
    """
    with open(prediction_path, "r") as fp:
        preds = fp.read().split("\n")
    if dataset is None:
        dataset = datasets.load_dataset(dataset_name, dataset_config_name, split=split)
    documents = dataset[text_column]
    labels = dataset[summary_column]
    labels = [
        l for d, l in zip(documents, labels) if d and l
    ]  # only keep the non-empty ones
    preds, labels = postprocess_text(preds, labels)
    metric = evaluate.load("bertscore")
    result = metric.compute(predictions=preds, references=labels, lang="en")
    result = {
        k: round(np.mean(v) * 100, 4) for k, v in result.items() if k != "hashcode"
    }
    return result


def get_latex_entries_for_rouge(a):
    """This function generates latex table entries from the rouge output."""
    a = list(a.values())[:2] + list(a.values())[3:]
    return " & ".join([f"{round(i, 2):.2f}" for i in a])


def get_latex_entries_for_reps(a):
    """This function generates latex table entries from the repetition output."""
    a = list(a.values())[:5]
    return " & ".join([f"{round(i, 2):.2f}" for i in a])


def get_rouge_result_in_list(
    prediction_path, dataset, text_column, summary_column, select_indices=None
):
    """
    Generates the rouge scores for each summary in the dataset and returns a list of rouge scores.

    This function is used for exploratory analysis of the generated summaries.
    """
    with open(prediction_path, "r") as fp:
        preds = fp.read().split("\n")
    documents = dataset[text_column]
    labels = dataset[summary_column]
    labels = [
        l for d, l in zip(documents, labels) if d and l
    ]  # only keep the non-empty ones
    preds, labels = postprocess_text(preds, labels)
    metric = evaluate.load("rouge")
    if select_indices is None:
        select_indices = range(len(preds))
    result = [
        metric.compute(predictions=[preds[i]], references=[labels[i]], use_stemmer=True)
        for i in select_indices
    ]
    return result


def process_docs_and_labels(dataset, text_column, summary_column):
    """This function is used to get the documents and the labels that match the predictions.

    This is needed when there are empty documents or labels in the dataset.

    The input dataset should be the test dataset.
    """

    # only keep the non-empty ones
    documents = dataset[text_column]
    labels = dataset[summary_column]
    documents, labels = zip(*[(d, l) for d, l in zip(documents, labels) if d and l])
    documents = list(documents)
    labels = list(labels)

    labels = [label.strip() for label in labels]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return documents, labels
