Modification of the cnn_dailymail dataset in Hugging Face. The main goal is to reproduce the results on BART. 

References: https://github.com/facebookresearch/fairseq/issues/1401

Major changes: 
1. remove the space in " ." in fix_missing_period.
2. remove "(CNN)" in article.