# FlowSUM

This repo contains the code for the EMNLP 2023 paper [Boosting Summarization with Normalizing Flows and Aggressive Training](https://openreview.net/forum?id=pGlnFVmI4x&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DEMNLP%2F2023%2FConference%2FAuthors%23your-submissions)). 

## Installation
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