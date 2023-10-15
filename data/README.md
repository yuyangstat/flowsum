# Datasets

This folder contains the data loading scripts for the datasets used. 
- `bart_cnndm` makes a slight modification to reproduce the BART result, following https://github.com/facebookresearch/fairseq/issues/1401. 
- `distil_cnndm` generates a CNN/DM dataset where the gold summaries in the training set is replaced with the pseudo-lables. 
    - Version 1.0.0 uses the pseudo-lables generated by facebook/bart-large-cnn.
    - Version 2.0.0 uses the pseudo-lables generated by sshleifer/pegasus-cnn-ft-v2.
- `distil_xsum` generates an XSum dataset where the gold summaries in the training set is replaced with the pseudo-lables.
    - Version 1.0.0 uses the pseudo-lables generated by google/pegasus-xsum.

These data loading scripts have been uploaded to Hugging Face Hub. Users can either load the datasets by running these data loading scripts locally, or directly use the Hugging Face interface as follows.

```python
import datasets
# bart_cnndm
data = datasets.load_dataset("yuyang/bart_cnndm", "3.0.0")

# distil_cnndm version 1.0.0
data = datasets.load_dataset("yuyang/distil_cnndm", "1.0.0")

# distil_cnndm version 2.0.0
data = datasets.load_dataset("yuyang/distil_cnndm", "2.0.0")

# distil_xsum version 1.0.0
data = datasets.load_dataset("yuyang/distil_xsum", "1.0.0")
```

Note that if users would like to modify the data loading scripts, please make sure the name of the folder and that of the script should always be the same.