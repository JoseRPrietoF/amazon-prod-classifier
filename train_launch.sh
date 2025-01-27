conda run --no-capture-output -n training python -m src.train.pipeline --config src/train/configs/bert_tiny_config.yaml --data_fraction=0.03
