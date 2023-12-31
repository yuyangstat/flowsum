export EXPERIMENT_NAME=example
mkdir -p ./checkpoints/$EXPERIMENT_NAME  # create saving directory if it doesn't exist: for saving log.
python ./train.py \
    --gpu_ids 0 1\
    --config_file_path ./experiments/test/$EXPERIMENT_NAME.yml \
    2>&1 | tee ./checkpoints/$EXPERIMENT_NAME/test.log
