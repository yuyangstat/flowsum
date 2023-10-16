# Trainers

This folder contains the custom trainers and callbacks.
- `callbacks.py` contains a visualization tensorboard callback, which write the visualization of latent distributions to TensorBoard.
- `trainer_bows_seq2seq.py` defines BOWsSeq2SeqTrainer. It has the following main features.
    1. It enables BOWs or avg_embed as additional input.
    2. It allows both standard VAE training and beta_C-VAE training. 
    3. It allows outputs containing perplexity, nf loss, and sentence generating time.
    4. It enables latent distribution visualization.
- `trainer_caat.py` builds on top of BOWsSeq2SeqTrainer and defines CAATBOWsSeq2SeqTrainer. It enables CAAT strategy.
- `trainer_time.py` applies to models from Hugging Face Hub. It allows outputing the sentence generating time.