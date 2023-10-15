# FlowSUM

This repo contains the code for the EMNLP 2023 paper [Boosting Summarization with Normalizing Flows and Aggressive Training](https://openreview.net/forum?id=pGlnFVmI4x&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DEMNLP%2F2023%2FConference%2FAuthors%23your-submissions)). 

## Installation
```python
# 1. create a virtual environment
conda create -n flowsum python=3.7

# 2. install PyTorch with GPU version
# choose the appropriate version that suits your laptop or server

# 3. install flowsum
python setup.py install
```

## Example
We provide a few example train/test configuration files in the `experiments` folder. To reproduce the results, check the hyper-parameter settings in Appendix G of the paper. 
```bash
bash train.sh

bash test.sh
```