# Fed dropout avg
python3 ./simulator.py --config-name fed_dropout_avg/cifar100.yaml ++fed_dropout_avg.round=1 ++fed_dropout_avg.epoch=1 ++fed_dropout_avg.worker_number=2 ++fed_dropout_avg.debug=True ++fed_dropout_avg.algorithm_kwargs.random_client_number=2

python3 ./simulator.py --config-name fed_paq/cifar100.yaml ++fed_paq.round=1 ++fed_paq.epoch=1 ++fed_paq.worker_number=2 ++fed_paq.debug=True ++fed_paq.algorithm_kwargs.random_client_number=2
