# CV
python3 ./simulator.py --config-name fed_avg/mnist.yaml ++fed_avg.round=1 ++fed_avg.epoch=1 ++fed_avg.worker_number=2 ++fed_avg.debug=True
# NLP
python3 ./simulator.py --config-name fed_avg/imdb.yaml ++fed_avg.round=1 ++fed_avg.epoch=1 ++fed_avg.worker_number=2 ++fed_avg.debug=True
# Graph
python3 ./simulator.py --config-name fed_gnn/cs.yaml ++fed_gnn.round=1 ++fed_gnn.epoch=1 ++fed_gnn.worker_number=2 ++fed_gnn.debug=False
# GTG
python3 ./simulator.py --config-name gtg_sv/mnist.yaml ++gtg_sv.round=1 ++gtg_sv.epoch=1 ++gtg_sv.worker_number=2 ++gtg_sv.debug=False
# OBD
python3 ./simulator.py --config-name fed_obd/cifar10.yaml ++fed_obd.round=2 ++fed_obd.epoch=1 ++fed_obd.worker_number=10 ++fed_obd.algorithm_kwargs.random_client_number=10 ++fed_obd.algorithm_kwargs.second_phase_epoch=1 ++fed_obd.debug=False
