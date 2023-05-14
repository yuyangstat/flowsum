export EXPERIMENT_NAME=example_aug
mkdir -p ./checkpoints/$EXPERIMENT_NAME  # create saving directory if it doesn't exist: for saving log.
python ./train.py \
    --gpu_ids 1 0\
    --config_file_path ./experiments/$EXPERIMENT_NAME.yml \
    2>&1 | tee ./checkpoints/$EXPERIMENT_NAME/train.log
