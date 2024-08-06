# distributed_learning_simulator

This is a simulator for distributed Machine Learning and Federated Learning on a single host. It implements common algorithms as well as our works.

## Installation

This is a Python project. The third party dependencies are listed in **requirements.txt**. If you use PIP, it should be easy to install:

```
python3 -m pip -r requirements.txt
```

## Our Works

### GTG-Shapley

To run the experiments of [GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning](https://dl.acm.org/doi/pdf/10.1145/3501811), use this command

```
bash gtg_shapley_train.sh
```

#### Reference

If you find our work useful, feel free to cite it:

```
@article{10.1145/3501811,
author = {Liu, Zelei and Chen, Yuanyuan and Yu, Han and Liu, Yang and Cui, Lizhen},
title = {GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning},
year = {2022},
issue_date = {August 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {13},
number = {4},
issn = {2157-6904},
url = {https://doi.org/10.1145/3501811},
doi = {10.1145/3501811},
journal = {ACM Trans. Intell. Syst. Technol.},
month = {may},
articleno = {60},
numpages = {21},
keywords = {Federated learning, contribution assessment, Shapley value}
}
```

### FedOBD

To run the experiments of [FedOBD: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning](https://arxiv.org/abs/2208.05174), use this command

```
bash fed_obd_train.sh
```

#### Reference

If you find our work useful, feel free to cite it:

```
@inproceedings{ijcai2023p394,
  title     = {FedOBD: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning},
  author    = {Chen, Yuanyuan and Chen, Zichen and Wu, Pengcheng and Yu, Han},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Edith Elkind},
  pages     = {3541--3549},
  year      = {2023},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2023/394},
  url       = {https://doi.org/10.24963/ijcai.2023/394},
}
```

### Historical Embedding-Guided Efficient Large-Scale Federated Graph Learning

The implementation has been move to other (GitHub repository)[https://github.com/cyyever/distributed_graph_learning_simulator]


#### Reference

If you find this work useful, feel free to cite it:

```
@article{li2024historical,
  title={Historical Embedding-Guided Efficient Large-Scale Federated Graph Learning},
  author={Li, Anran and Chen, Yuanyuan and Zhang, Jian and Cheng, Mingfei and Huang, Yihao and Wu, Yueming and Luu, Anh Tuan and Yu, Han},
  journal={Proceedings of the ACM on Management of Data},
  volume={2},
  number={3},
  pages={1--24},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```
