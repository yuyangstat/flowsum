# FlowSUM

This repo contains the code for the EMNLP 2023 paper [Boosting Summarization with Normalizing Flows and Aggressive Training](https://aclanthology.org/2023.emnlp-main.165/). 

## Setup
```bash
# 1. create a virtual environment
conda env create -f environment.yml

# 2. install PyTorch with GPU, the torch version we used was 1.12.1 and the CUDA version we used was 10.2
# choose the appropriate version that suits your laptop or server
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102

# 3. [Optional] install flowsum
python setup.py install
```

## Example
We provide a few example train/test configuration files in the `experiments` folder. To reproduce the results, check the hyper-parameter settings in Appendix G of the paper. 
```bash
bash train.sh

bash test.sh
```

## Citation
```bib
@inproceedings{yang-shen-2023-boosting,
    title = "Boosting Summarization with Normalizing Flows and Aggressive Training",
    author = "Yang, Yu  and
      Shen, Xiaotong",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.165",
    pages = "2727--2751",
    abstract = "This paper presents FlowSUM, a normalizing flows-based variational encoder-decoder framework for Transformer-based summarization. Our approach tackles two primary challenges in variational summarization: insufficient semantic information in latent representations and posterior collapse during training. To address these challenges, we employ normalizing flows to enable flexible latent posterior modeling, and we propose a controlled alternate aggressive training (CAAT) strategy with an improved gate mechanism. Experimental results show that FlowSUM significantly enhances the quality of generated summaries and unleashes the potential for knowledge distillation with minimal impact on inference time. Furthermore, we investigate the issue of posterior collapse in normalizing flows and analyze how the summary quality is affected by the training strategy, gate initialization, and the type and number of normalizing flows used, offering valuable insights for future research.",
}
```
