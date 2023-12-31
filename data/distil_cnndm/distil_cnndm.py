# coding=utf-8
# References:
# (1) https://huggingface.co/datasets/cnn_dailymail/blob/main/cnn_dailymail.py
# (2) https://huggingface.co/docs/datasets/dataset_script

"""Distilled CNN/DailyMail Summarization dataset."""

import os

import datasets
import nltk


_DESCRIPTION = """\
Distilled CNN/DailyMail non-anonymized summarization dataset.
There are two features:
  - article: text of news article, used as the document to be summarized
  - highlights: joined text of highlights with <s> and </s> around each
    highlight, which is the target summary

The pseudo labels are generated by running 
    1. facebook/bart-large-cnn on the CNN/DailyMail dataset, or
    2. sshleifer/pegasus-cnn-ft-v2 on the CNN/DailyMail dataset.

The files used here is directly downloaded from 
https://github.com/huggingface/transformers/blob/main/examples/research_projects/seq2seq-distillation/precomputed_pseudo_labels.md.
"""

_CITATION = ""

_DL_URLS = {
    "cnn_bart_pl": "https://cdn-datasets.huggingface.co/pseudo/cnn_dm/cnn_bart_pl.tgz",
    "cnn_pegasus_pl": "https://cdn-datasets.huggingface.co/pseudo/cnn_dm/pegasus_cnn_cnn_pls.tgz",
}

# as mentioned in https://github.com/huggingface/transformers/blob/main/examples/research_projects/seq2seq-distillation/precomputed_pseudo_labels.md#available-pseudo-labels,
#   about 5K are missing, and the training should be 282173.
_NUM_EXAMPLES = {"train": 282173, "val": 13368, "test": 11490}

# maps from datasets.Split to the one used in the downloaded data.
_SPLIT_MAP = {"train": "train", "test": "test", "validation": "val"}

_SUPPORTED_VERSIONS = [
    # Using the pseudo labels generated by BART.
    datasets.Version("1.0.0", "Using cased version and the one generated by BART."),
    # Using the pseudo labels generated by Pegasus.
    datasets.Version("2.0.0", "Using cased version and the one generated by PEGASUS."),
]

_DEFAULT_VERSION = datasets.Version("2.0.0", "Using cased version.")


class DistilCNNDMConfig(datasets.BuilderConfig):
    """BuilderConfig for DistilCNNDM."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DistilCNNDM(datasets.GeneratorBasedBuilder):
    """Distilled CNN/DailyMail non-anonymized summarization dataset."""

    BUILDER_CONFIGS = [
        DistilCNNDMConfig(name=str(version), description="Plain text", version=version)
        for version in _SUPPORTED_VERSIONS
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "article": datasets.Value("string"),
                    "highlights": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Split generators.

        Note that the validation data have prefix val instead of
            validation, so we use a split mapping.

        Although dl_manager is not used, we still need to keep it.
        """
        if self.config.version == "1.0.0":
            extracted_path = dl_manager.download_and_extract(_DL_URLS["cnn_bart_pl"])
            return [
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "src_path": os.path.join(
                            extracted_path,
                            "cnn_bart_pl",
                            f"{_SPLIT_MAP[split]}.source",
                        ),
                        "tgt_path": os.path.join(
                            extracted_path,
                            "cnn_bart_pl",
                            f"{_SPLIT_MAP[split]}.target",
                        ),
                        "num_examples": _NUM_EXAMPLES[_SPLIT_MAP[split]],
                    },
                )
                for split in [
                    datasets.Split.TRAIN,
                    datasets.Split.VALIDATION,
                    datasets.Split.TEST,
                ]
            ]
        elif self.config.version == "2.0.0":
            extracted_path = dl_manager.download_and_extract(_DL_URLS["cnn_pegasus_pl"])
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "src_path": os.path.join(
                            extracted_path,
                            "pegasus_cnn_cnn_pls",
                            "train.source",
                        ),
                        "tgt_path": os.path.join(
                            extracted_path,
                            "pegasus_cnn_cnn_pls",
                            "train.target",
                        ),
                        "num_examples": 287112,
                    },
                )
            ]

    def _generate_examples(self, src_path, tgt_path, num_examples):
        """This function returns the examples in the raw text form.

        The output article and highlights formats resemble those given
            by `load_dataset("cnn_dailymail", "3.0.0")`.
        """
        with open(src_path) as src, open(tgt_path) as tgt:
            for idx in range(num_examples):
                article = src.readline().strip()
                if article[:5] == "(CNN)":
                    article = article[5:]
                highlights = tgt.readline().strip()
                highlights = "\n".join(nltk.sent_tokenize(highlights))
                yield idx, {
                    "article": article,
                    "highlights": highlights,
                }
