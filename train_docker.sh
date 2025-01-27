path=./
OPTS=( -it );
OPTS+=( -v $(realpath ${path}):/app );
docker run --gpus all --ipc=host "${OPTS[@]}" amznclassif bash etl_launch.sh
# docker run --gpus all --ipc=host "${OPTS[@]}" amznclassif bash train_launch.sh
