# CV
python3 ./simulator.py --config-name fed_avg/mnist.yaml ++fed_avg.round=1 ++fed_avg.epoch=1 ++fed_avg.worker_number=2 ++fed_avg.debug=True
# NLP
python3 ./simulator.py --config-name fed_avg/imdb.yaml ++fed_avg.round=1 ++fed_avg.epoch=1 ++fed_avg.worker_number=2 ++fed_avg.debug=True
# Graph
python3 ./simulator.py --config-name fed_gnn/cs.yaml ++fed_gnn.round=1 ++fed_gnn.epoch=1 ++fed_gnn.worker_number=2 ++fed_gnn.debug=False
