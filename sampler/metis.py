from typing import Optional

import torch
import torch_geometric.typing
from torch import Tensor
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.sparse import index2ptr

from .base import SamplerBase


def METIS(data, num_parts: int, recursive: bool = False) -> Tensor:
    r"""Partitions a graph data object into multiple subgraphs""
    .. note::

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): If set to :obj:`True`, will use multilevel
            recursive bisection instead of multilevel k-way partitioning.
            (default: :obj:`False`)
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    # Computes a node-level partition assignment vector via METIS.

    # Calculate CSR representation:
    row, col = sort_edge_index(edge_index, num_nodes=num_nodes)
    rowptr = index2ptr(row, size=num_nodes)

    # Compute METIS partitioning:
    cluster: Optional[Tensor] = None

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        cluster = torch.ops.torch_sparse.partition(
            rowptr.cpu(),
            col.cpu(),
            None,
            num_parts,
            recursive,
        ).to(edge_index.device)
    if cluster is None and torch_geometric.typing.WITH_METIS:
        cluster = pyg_lib.partition.metis(
            rowptr.cpu(),
            col.cpu(),
            num_parts,
            recursive=recursive,
        ).to(edge_index.device)

    if cluster is None:
        raise ImportError("'METIS requires either 'pyg - lib' or 'torch - sparse'")

    return cluster


class GraphSplitter(SamplerBase):
    def __init__(self, config):
        dc = config.create_dataset_collection()
        METIS(dc.get_dataset_util().get_graph(0), num_parts=config.worker_number)
