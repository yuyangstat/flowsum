export PYTHONPATH="/home/yang6367/nf-summarizer/"
export EXPERIMENT_NAME=nf-bart-finetuned-rqnsf-lagtrain
mkdir -p ../checkpoints/$EXPERIMENT_NAME  # create saving directory if it doesn't exist: for saving log.
python ./train.py \
    --gpu_ids 4 5\
    --config_file_path ../experiments/nf_enhanced/train/$EXPERIMENT_NAME.yml \
    2>&1 | tee ../checkpoints/$EXPERIMENT_NAME/train.log

# python ./train.py \
#     --gpu_ids 5 4\
#     --config_file_path ../experiments/original/$EXPERIMENT_NAME.yml \
#     2>&1 | tee ../checkpoints/$EXPERIMENT_NAME/train.log    