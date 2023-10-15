# CNN/DailyMail Dataset for BART-based Models

This folder contains the data loading script for the CNN/DailyMail dataset, modified based on the [cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset in Hugging Face. The main goal is to reproduce the results on BART. It is also available in [Hugging Face Datasets](https://huggingface.co/datasets/yuyang/bart_cnndm).

Major changes: 
1. remove the space in " ." in fix_missing_period.
2. remove "(CNN)" in article.

## Usage
There are two ways of using this dataset. 

1. Directly load with the `datasets` library.
    ```python
    from datasets import load_dataset

    dataset = load_dataset("yuyang/bart_cnndm", "3.0.0")
    ```
2. Load the data with the data loading script locally. Note that after the following code is run once, we can then directly specify the `dataset_name` in the experiment configuration file to be "bart_cnndm".
    ```python
    from datasets import load_dataset

    load_dataset("path/to/bart_cnndm", "3.0.0")
    ```
## References
- https://github.com/facebookresearch/fairseq/issues/1401

