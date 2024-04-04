# fedaas
for configname in cs.yaml PubMed.yaml reddit.yaml yelp.yaml; do
  python3 ./simulator.py --config-name fed_aas/${configname} ++fed_aas.worker_number=50 ++fed_aas.round=50 ++fed_aas.model_name=TwoGATCN ++fed_aas.epoch=1 ++fed_aas.dataloader_kwargs.batch_number=10 ++fed_aas.learning_rate=0.001 ++fed_aas.algorithm_kwargs.edge_drop_rate=0.99 ++fed_aas.weight_decay=0.001 ++fed_aas.exp_name="fed_aas" ++fed_aas.algorithm_kwargs.min_sharing_interval=2
done
